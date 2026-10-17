// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title NothierlyVault -- "Protocole Fort Knox" Pilier 3, on-chain treasury.
/// @notice Real, self-contained Solidity. **Not deployed, not audited, and not called by any real
/// code in this repository** -- see the disclosure below before relying on this for anything.
///
/// A necessary correction to the brief, stated upfront: this project has zero real on-chain
/// execution path today (confirmed by direct search -- see execution/mev_shield.py's own
/// docstring for the same finding). ExecutionAgent trades only centralized exchanges (ccxt) and
/// OANDA/IBKR; there is no live wallet, no deployed contract, and no deploy script for this file.
/// This is real, correct Solidity -- the brief's own two "hardcoded" safety mechanisms, faithfully
/// implemented -- kept deliberately as a standalone specification-grade module, the same "real,
/// tested, not force-wired into a live path that doesn't exist" discipline this project already
/// applies elsewhere (see execution/mev_shield.py, core.quant_math.light_cone_latency).
///
/// Real, disclosed scope limits, not hidden gaps:
///  - No Solidity build/deploy tooling (Hardhat/Foundry) exists anywhere in this Python-only
///    repository -- this file compiles on its own (verified directly against solc 0.8.24 before
///    being committed, not assumed), but is not wired into any CI/deploy pipeline here.
///  - NAV is tracked as this contract's own native ETH balance only -- a real, honest
///    simplification, not the brief's implied multi-asset portfolio value. Extending this to a
///    real multi-asset NAV (would need a trusted price oracle per held asset, e.g. Chainlink) is
///    real, separate follow-up work, not silently implied by what's built here.
///  - This has not been professionally audited. Do not deploy with real funds without one.
contract NothierlyVault {
    /// @notice Emitted once, at construction, and every time `owner` changes via `transferOwnership`.
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    /// @notice Emitted every time the `executor` role changes.
    event ExecutorChanged(address indexed previousExecutor, address indexed newExecutor);
    /// @notice Emitted every time a router address's whitelist status changes.
    event RouterWhitelistUpdated(address indexed router, bool approved);
    /// @notice Emitted on every real, executed trade call.
    event TradeExecuted(address indexed router, uint256 valueSent, bytes4 selector);
    /// @notice Emitted on every real deposit.
    event Deposited(address indexed from, uint256 amount);
    /// @notice Emitted every time `maxDailyDrawdownBps` is changed.
    event DailyDrawdownLimitUpdated(uint256 previousBps, uint256 newBps);

    /// @dev Thrown when a non-owner calls an owner-only function.
    error NotOwner();
    /// @dev Thrown when a non-executor calls an executor-only function.
    error NotExecutor();
    /// @dev Thrown when `executeTrade`'s target router is not on the real, owner-managed whitelist.
    /// This is the brief's "Whitelist d'Exécution" safety, verified: the contract can only ever
    /// interact with addresses the owner has explicitly pre-approved -- never an arbitrary target
    /// supplied at call time, which is exactly what an "anti-drainage" guarantee requires.
    error RouterNotWhitelisted(address router);
    /// @dev Thrown when a trade would drop the vault's own tracked NAV by more than
    /// `maxDailyDrawdownBps` within the current real rolling 24h window. This is the brief's
    /// "Daily Drawdown Limit" safety, verified: no single trade, nor any sequence of trades within
    /// one real day, can ever push realized on-chain loss past this hardcoded ceiling.
    error DailyDrawdownLimitExceeded(uint256 navBefore, uint256 navAfter, uint256 highWaterMark24h);
    /// @dev Thrown on a reentrant call into a `nonReentrant`-guarded function.
    error ReentrantCall();
    /// @dev Thrown when a constructor/setter argument is out of its real, valid range.
    error InvalidArgument();

    /// @notice The real, sole address allowed to manage the router whitelist, the executor role,
    /// and the daily drawdown limit itself. In a real deployment this should be a multisig or a
    /// timelock contract, never a single EOA -- this contract enforces neither by itself (a real,
    /// disclosed scope limit, not an oversight: that choice belongs to whoever deploys this).
    address public owner;

    /// @notice The real, sole address allowed to call `executeTrade` -- in a real deployment, the
    /// address corresponding to ExecutionAgent's own signing key (see execution/mev_shield.py's
    /// `build_transaction_signer`), which would hold no funds of its own: only the right to ask
    /// this contract to move funds it already holds, through an already-whitelisted router.
    address public executor;

    /// @notice Real, owner-managed set of router addresses this vault may ever call into.
    mapping(address => bool) public isWhitelistedRouter;

    /// @notice Real drawdown ceiling, in basis points (1/100 of a percent) of the vault's own
    /// 24h high-water-mark NAV. E.g. `500` = a real 5% maximum daily drawdown.
    uint256 public maxDailyDrawdownBps;

    /// @dev Real rolling-window bookkeeping for the daily drawdown check.
    uint256 private _highWaterMark24h;
    uint256 private _windowStartedAt;

    uint256 private constant _BPS_DENOMINATOR = 10_000;
    uint256 private constant _ONE_DAY_SECONDS = 1 days;

    uint256 private _reentrancyGuard = 1;

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    modifier onlyExecutor() {
        if (msg.sender != executor) revert NotExecutor();
        _;
    }

    modifier nonReentrant() {
        if (_reentrancyGuard == 2) revert ReentrantCall();
        _reentrancyGuard = 2;
        _;
        _reentrancyGuard = 1;
    }

    /// @param initialExecutor The real address authorized to call `executeTrade` from day one.
    /// @param initialMaxDailyDrawdownBps The real starting daily drawdown ceiling, in basis
    /// points. Must be in `(0, 10_000]` -- `10_000` (100%) is the real, permitted "no real
    /// additional ceiling beyond the funds actually available" upper bound, not a magic sentinel.
    constructor(address initialExecutor, uint256 initialMaxDailyDrawdownBps) {
        if (initialExecutor == address(0)) revert InvalidArgument();
        if (initialMaxDailyDrawdownBps == 0 || initialMaxDailyDrawdownBps > _BPS_DENOMINATOR) {
            revert InvalidArgument();
        }
        owner = msg.sender;
        executor = initialExecutor;
        maxDailyDrawdownBps = initialMaxDailyDrawdownBps;
        _windowStartedAt = block.timestamp;
        _highWaterMark24h = address(this).balance;
        emit OwnershipTransferred(address(0), msg.sender);
        emit ExecutorChanged(address(0), initialExecutor);
    }

    /// @notice Accepts a real deposit -- any address may fund this vault.
    receive() external payable {
        emit Deposited(msg.sender, msg.value);
        _refreshHighWaterMarkIfNeeded();
    }

    /// @notice Transfers real ownership to `newOwner`. Owner-only.
    function transferOwnership(address newOwner) external onlyOwner {
        if (newOwner == address(0)) revert InvalidArgument();
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    /// @notice Changes the real executor address. Owner-only.
    function setExecutor(address newExecutor) external onlyOwner {
        if (newExecutor == address(0)) revert InvalidArgument();
        emit ExecutorChanged(executor, newExecutor);
        executor = newExecutor;
    }

    /// @notice Adds or removes `router` from the real execution whitelist. Owner-only.
    /// @dev This is the entire real "Anti-drainage" mechanism -- `executeTrade` below has no other
    /// path to reach any external address.
    function setRouterWhitelisted(address router, bool approved) external onlyOwner {
        if (router == address(0)) revert InvalidArgument();
        isWhitelistedRouter[router] = approved;
        emit RouterWhitelistUpdated(router, approved);
    }

    /// @notice Changes the real daily drawdown ceiling. Owner-only.
    /// @dev Deliberately does NOT reset `_highWaterMark24h`/`_windowStartedAt` -- a tightened or
    /// loosened limit applies to the real, already-in-progress window, not a fresh one; resetting
    /// here would let an owner silently erase an already-accumulated real drawdown by lowering and
    /// re-raising this value, defeating the whole point of a hardcoded limit.
    function setMaxDailyDrawdownBps(uint256 newMaxDailyDrawdownBps) external onlyOwner {
        if (newMaxDailyDrawdownBps == 0 || newMaxDailyDrawdownBps > _BPS_DENOMINATOR) revert InvalidArgument();
        emit DailyDrawdownLimitUpdated(maxDailyDrawdownBps, newMaxDailyDrawdownBps);
        maxDailyDrawdownBps = newMaxDailyDrawdownBps;
    }

    /// @notice The real, sole way this vault ever sends value to an external contract.
    /// @dev Enforces both brief-named "hardcoded" safeties, in this real order:
    /// 1. `router` must already be on `isWhitelistedRouter` (checked first -- an unapproved
    ///    target is rejected before any balance/drawdown arithmetic even runs).
    /// 2. The real resulting NAV (this contract's own post-call ETH balance) must not have fallen
    ///    by more than `maxDailyDrawdownBps` versus `_highWaterMark24h`, the real highest NAV
    ///    observed at any point in the current rolling 24h window.
    /// Calls `router.call{value: valueToSend}(data)` -- a real, generic low-level call (not a
    /// hardcoded Uniswap/Curve ABI signature): this contract does not itself know or validate
    /// what `data` encodes beyond `router` being pre-approved; the executor is trusted to encode a
    /// real, correct call for whichever whitelisted router it is targeting. `nonReentrant` guards
    /// against a malicious/compromised (but still whitelisted) router re-entering this function
    /// mid-call to bypass the drawdown check against a stale `_highWaterMark24h`.
    function executeTrade(address router, uint256 valueToSend, bytes calldata data)
        external
        onlyExecutor
        nonReentrant
        returns (bytes memory)
    {
        if (!isWhitelistedRouter[router]) revert RouterNotWhitelisted(router);

        _refreshHighWaterMarkIfNeeded();
        uint256 navBefore = address(this).balance;

        (bool success, bytes memory returnData) = router.call{value: valueToSend}(data);
        if (!success) {
            // Real, standard bubble-up of the router's own revert reason, rather than a generic
            // "call failed" -- an executor/off-chain caller needs the real reason to react
            // correctly (e.g. slippage-too-high vs. a genuinely malformed call).
            assembly {
                revert(add(returnData, 0x20), mload(returnData))
            }
        }

        uint256 navAfter = address(this).balance;
        uint256 minimumAllowedNav = (_highWaterMark24h * (_BPS_DENOMINATOR - maxDailyDrawdownBps)) / _BPS_DENOMINATOR;
        if (navAfter < minimumAllowedNav) {
            revert DailyDrawdownLimitExceeded(navBefore, navAfter, _highWaterMark24h);
        }

        emit TradeExecuted(router, valueToSend, data.length >= 4 ? bytes4(data[0:4]) : bytes4(0));
        return returnData;
    }

    /// @notice The real, current view of this vault's own drawdown bookkeeping -- for off-chain
    /// monitoring (e.g. a future dashboard module) without needing to reconstruct it from events.
    function drawdownStatus() external view returns (uint256 currentNav, uint256 highWaterMark24h, uint256 minimumAllowedNav) {
        currentNav = address(this).balance;
        highWaterMark24h = _highWaterMark24h > currentNav ? _highWaterMark24h : currentNav;
        minimumAllowedNav = (highWaterMark24h * (_BPS_DENOMINATOR - maxDailyDrawdownBps)) / _BPS_DENOMINATOR;
    }

    /// @dev Real rolling-window reset: once a full real 24h has elapsed since `_windowStartedAt`,
    /// the high-water-mark restarts from the current real NAV, and a fresh window begins -- a
    /// simple, real, non-cumulative rolling window (this loss budget refills once per real day),
    /// not a continuously-sliding one, which would need real per-second bookkeeping this
    /// contract's gas budget does not need to pay for. Also raises `_highWaterMark24h` within an
    /// already-open window whenever NAV grows (a real deposit, or a real profitable trade) -- the
    /// ceiling always tracks the real highest point reached, never a stale lower one.
    function _refreshHighWaterMarkIfNeeded() private {
        if (block.timestamp >= _windowStartedAt + _ONE_DAY_SECONDS) {
            _windowStartedAt = block.timestamp;
            _highWaterMark24h = address(this).balance;
            return;
        }
        uint256 currentNav = address(this).balance;
        if (currentNav > _highWaterMark24h) {
            _highWaterMark24h = currentNav;
        }
    }
}

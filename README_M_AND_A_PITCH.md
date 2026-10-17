# Nothierly — Note de structure pour due diligence institutionnelle (M&A)

**Ce document est une note d'ingénierie interne, pas un mémorandum d'offre.** Il décrit honnêtement
ce qui existe dans ce dépôt et à quel stade de maturité, avec la même discipline de vérification
que celle déjà appliquée dans [`nothierly_genesis/TECH_ATTRIBUTION_REPORT.md`](nothierly_genesis/TECH_ATTRIBUTION_REPORT.md)
et [`nothierly_genesis/README.md`](nothierly_genesis/README.md) : chaque affirmation de statut est
vérifiable par `grep`/lecture directe du code, pas supposée. Toute levée de fonds réelle nécessite un
audit financier et juridique indépendant que ce document ne remplace pas.

## 1. Ce qu'est réellement Nothierly aujourd'hui

Un système d'exécution multi-agents pour l'arbitrage crypto/P2P/multi-actifs, organisé en trois
couches :

- **Un EMS (Execution Management System) opérationnel** : `src/dashboard/nothierly_ems.py`, une
  application Streamlit multi-pages avec authentification réelle (`core/iam.py` : comptes par
  utilisateur, hachage Argon2, 2FA TOTP optionnelle, révocation de session, verrouillage d'inactivité
  — voir le docstring de `nothierly_ems.py` pour le détail et ses limites honnêtes).
- **Un moteur d'exécution réel** : `core/crossing_network.py`, le moteur d'internalisation
  effectivement utilisé en production (voir la table de statut de `TECH_ATTRIBUTION_REPORT.md`),
  entouré de modules de risque réels (`core/circuit_breaker.py`, `risk/cross_margin_engine.py`,
  `risk/regime_detector.py`) et d'une infrastructure SRE réelle (`core/exception_manager.py`,
  `core/watchdog.py`).
- **Une division R&D isolée par conception** (`nothierly_genesis/`) : dépendances lourdes
  (`torch`, `ripser`, `dowhy`) volontairement tenues à l'écart du chemin de production
  latency-sensitive, chaque résultat scoré par un statut de validation explicite.

## 2. Statut réel des modules "frontier tech" du dashboard

Chaque page "exotique" de l'EMS (`src/dashboard/ems_pages/singularity_engine.py`,
`institutional_audit.py`, `black_swan_lab.py`, `gauge_theory_visualizer.py`,
`quantum_inspired_routing.py`, `cyber_defense_core.py`, `non_ergodic_physics.py`,
`quantum_entanglement.py`, `apex_leviathan.py`, `wolfram_hypergraph.py`) applique la **même règle,
sans exception** : un bandeau de transparence en tête de page distingue ce qui est un vrai calcul
(exécuté en direct, pas câblé en dur) de ce qui est une visualisation conceptuelle ou artistique, et
aucune ne présente un chiffre de performance, de sécurité ou de conformité inventé comme s'il était
un résultat réel. Le détail module par module (ce qui est vérifié, ce qui ne l'est pas encore) vit
dans le docstring de chaque fichier — c'est la source de vérité, pas un résumé marketing séparé qui
pourrait diverger du code.

Pour la division `nothierly_genesis/` spécifiquement (Topological Crash Detector, Spiking Tick
Network, Causal Impact Engine, Quantum Batch Matcher, ZK Matching Engine, TDA Engine), le statut
d'intégration réel — quels modules touchent un ordre réel, lesquels sont R&D pure — est tenu à jour
dans [`TECH_ATTRIBUTION_REPORT.md`](nothierly_genesis/TECH_ATTRIBUTION_REPORT.md), reconstruit par
`grep` sur tout le pipeline de production, pas par déclaration.

## 3. Ce qui a été refusé — et pourquoi c'est un signal de gouvernance, pas une lacune

Trois demandes de fonctionnalités ont été refusées explicitement, y compris comme simple simulation
d'interface, parce qu'elles décrivent des techniques réelles et dommageables plutôt que de la
R&D dramatisée :

| Demandé | Refusé parce que | Où |
|---|---|---|
| "Ghost Network" — éclater un ordre en micro-transactions maquillées pour échapper à la détection AML | Décrit du *structuring* réel, un délit financier réel (31 U.S.C. § 5324 et équivalents) | `black_swan_lab.py` |
| Trade à latence zéro via intrication quantique | Contredit le théorème de non-communication, un résultat établi de la théorie de l'information quantique, pas une limite d'ingénierie actuelle | `quantum_entanglement.py` |
| "Vol de compute" via WebAssembly dans le navigateur de tiers | Décrit du cryptojacking navigateur, une catégorie de malware réelle et un accès non autorisé (18 U.S.C. § 1030 et équivalents) | `apex_leviathan.py` |

Pour un investisseur institutionnel qui va, de toute façon, faire vérifier chaque affirmation
technique par ses propres experts avant de signer, un système qui refuse par construction de
fabriquer des capacités ou des métriques est un actif de confiance plus solide qu'une démo qui
impressionne au premier coup d'œil et s'effondre à la première question de due diligence.

## 4. Ce que ce document ne fait pas

Il ne prétend à aucun chiffre de performance de trading en capital réel, à aucune certification de
sécurité obtenue, à aucune intégration tierce (KMS multi-parties, scoring wallet type Chainalysis,
FHE en production) qui n'existe pas dans ce dépôt aujourd'hui. Pour la liste précise de ce qui est
vérifié versus conceptuel sur chaque page, se référer au docstring du fichier concerné — il est
tenu à jour avec le code, ce qu'un document séparé ne peut pas garantir dans la durée.

# Rapport d'attribution technologique — modules "exotiques"

**Question posée** : ces modules apportent-ils un edge réel (P&L, risque, ou avantage produit démontrable), ou sont-ils de la R&D non intégrée ? Réponse mesurée, pas supposée — méthode identique à celle déjà établie par `nothierly_genesis/README.md`.

## Statut d'intégration réel (vérifié par grep sur tout le pipeline de production)

| Module | Fichier | Utilisé par un agent/moteur live ? | Preuve |
|---|---|---|---|
| Topological Crash Detector | `nothierly_genesis/topological_crash_detector.py` | ❌ Non — bouton dashboard "Tier 2 R&D" uniquement | `agent_fleet.py` l.260 (exécution manuelle à la demande) |
| Spiking Tick Network | `nothierly_genesis/spiking_tick_network.py` | ❌ Non — idem | `agent_fleet.py` l.256-260 |
| Causal Impact Engine | `nothierly_genesis/causal_impact_engine.py` | ❌ Non — idem | `agent_fleet.py` l.291-295 |
| **Quantum Batch Matcher** | `core/quantum_batch_matcher.py` | ❌ Non — aucune référence hors de son propre fichier | `grep` : 0 appelant réel |
| **ZK Matching Engine** | `crypto/zk_matching_engine.py` | ❌ Non — aucune référence nulle part, même pas au dashboard | `grep` : 0 appelant, y compris dans les tests d'intégration |
| *(Référence)* Crossing Network (plaintext) | `core/crossing_network.py` | ✅ **Oui** — c'est LE moteur d'internalisation réellement utilisé | `shadow_crossing_engine.py`, `internal_dark_pool.py`, `fix_server.py` |

**Conclusion brute** : aucun des 5 modules "frontier tech" ne touche un seul ordre réel aujourd'hui. Le moteur de crossing qui tourne réellement en production (`ShadowCrossingEngine`, enregistré dans `run_backend.py`) utilise la version plaintext (`core/crossing_network.py`), pas ZK, pas quantique.

## Verdict par module (0 = pure R&D, sans impact ; 1 = intégrable tel quel ; 2 = déjà production-adjacent)

| Module | Solidité technique | Verdict d'apport réel | Recommandation |
|---|---|---|---|
| Quantum Batch Matcher | Correct, mathématiquement vérifié (QUBO validé contre force brute) | **Auto-invalidé par son propre benchmark** : à N=100-1000 ordres (échelle réelle d'un dark pool), un simple algorithme glouton est 1000-11000× plus rapide ET trouve un meilleur résultat (voir docstring du module, mesures déjà faites). L'explication est honnête : le problème a une structure bipartite qu'un solveur QUBO générique ne peut pas exploiter. | **Ne pas intégrer tel quel.** Le garder comme preuve de compétence R&D documentée, pas comme composant produit. |
| ZK Matching Engine | Cryptographie réelle et correcte (Pedersen + Chaum-Pedersen sur BN128, vérifiée contre de la vraie arithmétique de courbe) | **Seul module dont la propriété de valeur (confidentialité de la taille/du sens d'un ordre) est réelle ET non invalidée par un test.** Limite honnête déjà documentée : seuls les crosses exactement compensés sont prouvés (pas de fill partiel), et l'opérateur voit quand même les deux côtés au moment du match (pas de MPC). | **Seul candidat sérieux à l'intégration.** Voir plus bas. |
| Topological / Spiking / Causal (nothierly_genesis) | Architecture-verified selon leur propre README | Explicitement non calibrés sur un vrai crash historique, jamais tournés en Shadow Mode (aveu direct du README) | Suivre le plan déjà écrit dans le README section "Path to production" — étape 1 : incubation en Shadow Mode, pas encore faite. |

## Mise à jour -- corrections appliquées

**ZK Matching Engine, maintenant réellement branché.** `src/execution/shadow_crossing_engine.py` rejoue désormais chaque match réel observé à travers `crypto/zk_matching_engine.py`'s réel pipeline commit/open/proof/verify, et vérifie publiquement le résultat une seconde fois, indépendamment. Portée honnête : ceci ne rend pas l'observation de `ShadowCrossingEngine` elle-même confidentielle (elle lit déjà la quantité/le sens en clair depuis `CLIENT_ORDERS_TOPIC`) -- ce que ça démontre, pour la première fois sur du flux d'ordres réel et non plus seulement synthétique, c'est que le pipeline cryptographique produit un règlement correct et vérifiable publiquement, sans révéler la taille ni le sens à un tiers. 9 tests réels (dont un test de rejet de preuve falsifiée), mypy propre. Jamais branché dans `core.crossing_network` (qui gère de vrais ordres clients) -- seulement dans le moteur shadow, sans capital réel, cohérent avec la limite de prudence posée initialement.

**Quantum Batch Matcher, maintenant explicitement et irrévocablement déprécié.** Chaque appel à `solve_batch_selection` émet désormais un vrai `DeprecationWarning` au runtime (pas seulement une note dans le docstring) pointant vers `solve_batch_selection_greedy`, l'alternative déjà prouvée meilleure par le propre benchmark du module. Empêche qu'un futur appelant l'intègre en production sans voir pourquoi il ne devrait pas.

**Nouveau module ajouté, même statut que les trois autres "nothierly_genesis" ci-dessus : `nothierly_genesis/tda_engine.py` (+ `plot_persistence.py`).** Détection de cycles d'arbitrage réels par homologie persistante (ripser, filtration de Vietoris-Rips) sur une matrice de distance géodésique dérivée d'un marché P2P simulé à 6 nœuds. **Correction mathématique réelle appliquée avant même d'écrire le code** : la formule brute `-ln(taux)` du brief ne donne pas une distance de Rips valide (signe/échelle arbitraires par devise, asymétrie directionnelle) -- remplacée par une distance géodésique construite sur des multiplicateurs de valeur normalisés, avec Bellman-Ford (`networkx.find_negative_cycle`) comme vérité terrain rigoureuse en parallèle de la couche TDA. **Bug réel trouvé en exécutant le module une première fois** (pas supposé, observé) : la première topologie plantait un cycle négatif plus court et non voulu en plus du cycle à 4 pattes prévu -- corrigé en retravaillant la topologie du marché, revérifié : Bellman-Ford retrouve maintenant exactement +8.000%, et bêta_1=1 est confirmé indépendamment par ripser. **Limite honnête** : identique aux trois autres modules de ce tableau -- architecture vérifiée sur un cas synthétique planté, jamais calibré sur un vrai flux d'arbitrage historique, non branché à un agent live (script autonome uniquement, `python -m nothierly_genesis.tda_engine`). 7 tests réels, mypy propre.

**Rien de tout ceci n'a interrompu le shadow run en cours** : aucun des fichiers modifiés n'est chargé par les processus actifs (fakeredis, market-data-agent, alpha-engine) -- `ShadowCrossingEngine` et `quantum_batch_matcher` ne font partie que du fleet complet (`scripts/run_backend.py`), non démarré dans cette session.

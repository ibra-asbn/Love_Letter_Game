Step 3 de Love Letter : pourquoi ça flotte et que faire
Diagnostic en une ligne : votre signal stratégique est réel (le teacher rollout-guided gagne
+2.29 pts en arena), mais vous le distillez mal pour trois raisons additives — labels MC trop
bruités au regard de la résolution recherchée, objet appris (Q absolu) sub-optimal
pour un problème de classement, et mismatch de distribution offline entre le teacher et
le ranker en inférence. Aucune des trois ne demande de réécrire le pipeline. Les trois
corrections principales (variance reduction par CRN, tête advantage dueling, filtrage
statistique des paires) sont multiplicatives et tiennent en quelques jours sur Mac Pro. À
court terme, n’abandonnez pas la distillation locale — vous n’avez pas encore essayé les
techniques qui marchent dans la littérature analogue (AlphaZero, ExIt, Tesauro rollouts,
Dueling, AWR). Le pivot vers CFR/PoG/ReBeL serait prématuré : la littérature 2024-2026
(Rudolph, Sokota et al., ICLR 2026) montre qu’un PG bien tuné égale les méthodes CFR
profondes sur les benchmarks d’exploitabilité, et CFR perd ses garanties à 4 joueurs de
toute façon.
Mais — et c’est le point dur — si après application de ces correctifs vous restez bloqué,
le bottleneck n’est pas représentationnel, c’est de la recherche. Auquel cas la bonne
pivot n’est pas CFR mais garder un peu de recherche en inférence : MO-IS-MCTS avec
votre ranker comme prior PUCT et déterminisations échantillonnées depuis votre belief net.
C’est la conclusion convergente de toutes les analyses ci-dessous.
1. Le calcul de bruit qui change tout
À 16-32 rollouts/action sur récompense binaire/discrète, l’écart-type empirique d’une
moyenne MC est σ ≈ 0.5/√N ≈ 0.09–0.13 sur [-1,1]. L’intervalle de Wilson 95% à N=16,
p̂ =0.5 fait ±0.24 ; à N=32, ±0.17. arxiv Vos écarts typiques entre actions candidates sont
souvent < 0.10, donc sous le plancher de bruit. Conclusion : vos paires (s, a_better,
a
_worse) construites sur des écarts type 9/16 vs 10/16 sont statistiquement
indistinguables — le ranker apprend littéralement du bruit sur ces paires. C’est exactement
ce que dit votre diagnostic « ranker apprend trop souvent des écarts faibles qui sont du
bruit ». La littérature MCTS (AlphaGo Zero, ExIt, Polygames, Soemers et al. 2024 sur Wilson
en general game playing) y répond par trois mécanismes que vous n’utilisez pas encore :
distillation des visit counts plutôt que des Q, filtrage par intervalle de confiance avant
labellisation, et surtout — point crucial — réduction de variance par randomness sharing.
2. Les trois corrections à appliquer dans cet ordre
2.1 Common Random Numbers (CRN) / rollouts appariés — le plus gros
levier, à faire en premier
Idée : quand vous comparez les actions candidates a₁…a
_K depuis l’état s,
n’échantillonnez pas K déterminisations indépendantes. Échantillonnez une
déterminisation D et un seul stream de RNG ω, puis jouez les K rollouts avec ce même (D,
ω) — seule la première action change. Tesauro (Backgammon, 2002) le formule clairement :
« les erreurs systématiques dans les scores rollout des coups frères sont fortement
corrélées et s’annulent dans la différence ». Brynmawr Glasserman & Yao (1992) prouvent
que sous des hypothèses de monotonie/continuité faibles, la variance de la différence Q̂ (a)
−Q̂ (a’) chute de Var(X)+Var(X’)−2Cov(X,X’). En jeux de cartes, Cov est typiquement très
grand positif → réduction de variance 5×–20× sur les jeux de cartes (Bjarnason 2009 sur
Klondike, Cowling/Powley/Whitehouse 2012 sur ISMCTS).
Recette concrète :
pour r in range(N_rollouts): # 16 suffit
D = sample_determinization(s) # PARTAGÉ
seed_ω = base_seed + r # PARTAGÉ
pour a in candidate_actions:
env = clone(s, D, seed=seed_ω)
env.step(a)
ret_a[r] = playout(env, opponent_policy=Step2)
# Confiance par t-test PAIRÉ (pas Welch !)
diff_r = ret_a[r] - ret_a'[r]
t = mean(diff_r) / (std(diff_r) / sqrt(N))
Le seul bug à éviter : votre simulateur Love Letter doit être déterministe étant donné (état,
seed). Tout random.random() ou np.random.choice éparpillé dans le code casse le CRN.
Passez un objet RNG explicitement à chaque fonction stochastique. Sanity check : la
corrélation empirique ρ entre rollouts appariés doit être >0.5 (visez 0.7-0.8) sur un
échantillon ; si ρ<0.3, votre plomberie est cassée.
Effet attendu : passe de « 16 rollouts non-appariés indistinguables » à « 16 rollouts
appariés ≈ 80 rollouts non-appariés effectifs ». La fraction de paires (s, a, a’)
statistiquement séparables explose, et avec elle la qualité des labels.
2.2 Tête advantage dueling + perte pairwise margin-confidence — deuxième
priorité
Pourquoi pas Q absolu : Q(s,a) absorbe d’énormes composantes state-dependent («
quelle est la qualité de ma main, du board ») qui sont identiques pour toutes les actions.
Régresser sur Q force le réseau à mémoriser V(s) plusieurs fois. Le signal utile pour le
classement des actions est A(s,a) = Q(s,a) − V(s). C’est précisément l’argument de Wang
et al. 2016 (Dueling Network Architectures) qui notent que la décomposition aide
spécifiquement quand « beaucoup d’actions ont des valeurs similaires » ResearchGate +2 —
le cas Love Letter.
Architecture (dueling avec centrage, sinon V et A sont sous-déterminés) :
trunk(obs, hidden_step2, belief) → V(s) ∈ ℝ
Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)
→ A(s,a) ∈ ℝ^|A_legal|
Loss combinée :
λ
_
V · MSE(V̂ (s), z
_outcome)
λ
_
A · pairwise_loss(A(s,·), preference_pairs)
petit weight decay sur A pour pousser A≈0 quand non-discriminant
Loss pairwise margin-aware (clé pour la calibration) :
m
ij = max(0, |Δp̂
ij| − ε
noise) _
_
_
L = -log σ((A(s,a⁺) − A(s,a⁻))/β − m_ij)
# marge ∝ confiance MC
Les paires confiantes (Δp̂ grand) supplient un fort gradient, les paires ambiguës un faible —
c’est l’équivalent AWR/CRR (Peng et al. 2019, arXiv Wang et al. 2020) appliqué à votre
setting. arXiv Transferlab Hyperparamètres typiques : β = 0.5–1.0, λ
V:λ
_
_
A = 0.5:1.0,
weight decay 1e-4.
Pourquoi cela soigne aussi votre problème de seuil d’override : un argmax sur A est
invariant aux erreurs sur V. Votre problème actuel — « marges prédites mal calibrées,
classement local correct mais score pas fiable comme seuil » — vient en partie de ce que
vous régressez sur des Q absolus dont la dynamique est dominée par V. Avec une tête A
centrée, l’amplitude de A est typiquement [-0.3, 0.3] dans des jeux de ce type, beaucoup
plus stable à seuiller.
2.3 Filtrage statistique des paires — gardien anti-bruit
Construisez des paires (s, a⁺ , a⁻) seulement quand :
1. min(n_a, n_a’) ≥ 8 rollouts
2. Welch ou paired-t p < 0.10 (loose, vous avez beaucoup d’états)
3. |Δp̂ | ≥ 0.10 plancher d’effet
Les paires qui échouent (1)+(2) ne disparaissent pas : utilisez-les comme paires « tie »
dans une perte symétrique (régulariseur), pas comme préférence dure. Avec CRN, utilisez la
variance pairée dans le test t — c’est le changement code le plus important qui fait passer
~30-50% des paires de « bruit » à « signal exploitable ».
Cible empirique : après filtrage, ~30-60% de vos paires brutes doivent passer. Si >90%,
gate trop lâche. Si <10%, trop strict. Sur les paires retenues, votre validation pair-accuracy
doit pouvoir dépasser 80-85% au lieu de plafonner à ~60-70% comme actuellement.
Astuce sequential / SPRT : ne lancez pas 16 rollouts d’office. Démarrez avec 4 par action,
et n’allouez de nouveaux rollouts qu’aux paires où le test paired-t n’est pas encore
concluant (|t|<2). Souvent multiplie le nombre d’états labellisables par 1.5-2× à compute
égal.
3. Boucle DAgger légère pour fermer le mismatch — quatrième
priorité
Le diagnostic 3 de votre liste (« ranker visite distribution différente, pas de DAgger ») est
correct et bien identifié dans la littérature. La théorie Ross-Bagnell (AISTATS 2010, 2011)
garantit O(T) au lieu de O(T²) erreur composée, OpenReview Emergent Mind et 3 à 5
itérations DAgger suffisent en pratique dans la plupart des projets (SuperTuxKart 5,
Mario 15-20, ThriftyDAgger 3, robotique 3-10). À noter qu’ExIt formalise AlphaZero
comme une instance de DAgger : à chaque cycle, le teacher (= MCTS guidé par l’apprenti
courant) génère des cibles, l’apprenti se met à jour, et le teacher s’améliore
automatiquement NeurIPS — c’est exactement votre setup si votre teacher rollout utilise le
ranker comme rollout policy.
Recette minimum-viable adaptée à un teacher coûteux : ThriftyDAgger soft + cibles
ExIt-style.
N_ITER = 3–5
GAMES_PER_ITER = 400 QUERY_BUDGET = 1500/iter TEMP_SOFT = 1.0 ALPHA_REPLAY = 0.5 # ~6000 décisions self-play
# gating: query teacher seulement si :
# margin(top1, top2) < 0.15
# OU disagreement ensemble > τ
# OU novelty > 0.85 (distance latente)
# cible = softmax(Q/τ), pas argmax
# 50% nouveau / 50% replay stratifié par phase
β = 0 dès iter 1 # Ross 2011 standard
Trois choix critiques :
Cibles soft (Q-vector ou KL sur softmax(Q/τ)), pas 1-hot. Multiplicateur 3-10× sur la
valeur d’information par query coûteuse — c’est l’enseignement central d’AlphaZero
(visit counts comme cible faible-variance) et de ExIt.
Stratification du replay par phase de partie (deck progression). Vos résultats sur
stage-weighting montrent que la signal/noise dépend fortement de la phase ; ne laissez
pas les états de fin de manche (rares mais critiques) être noyés par les états de début.
Fine-tune, pas re-train from scratch à chaque itération.
Plan B si le pipeline async vous semble lourd : DART (Laskey et al., CoRL 2017). Injectez
du bruit (epsilon-greedy sur top-2, ou softmax avec température) dans le teacher pendant
la collecte, pour qu’il visite des états off-policy et apprenne à s’en sortir. arXiv +3 Aucune
boucle online, pas d’asynchrone. Empiriquement DART récupère ~80% du gain DAgger en
MuJoCo pour 3× moins de compute.
Spécifique Mac Pro : votre bottleneck est le simulateur, pas le NN. Optimisez la sim Love
Letter en pur Python avec tableaux numpy compacts (ou Numba/Cython si vous avez 2
jours), et parallélisez via multiprocessing (pas threading — GIL). Sur M2 Ultra (16-24
cores), 16-20 workers self-play + 4-8 workers teacher en pipeline producer-consumer →
~5-10 min wall-clock par itération de labellisation.
4. La pivot pertinente n’est pas CFR — c’est garder un peu de
recherche en inférence
Si après §2 et §3 vous restez bloqué, le diagnostic ne sera pas « il faut plus de capacité
» mais « la skill du teacher vient de la recherche qu’un single forward pass ne peut
compresser ». C’est très facile à diagnostiquer (voir §6) — faites un balayage du budget de
recherche du teacher : si la courbe de win-rate vs B (=1, 4, 16, 64, 256) sature vite (B≤4),
votre ranker peut absorber le signal ; si elle continue à monter à B=64, vous demandez
l’impossible à un MLP.
Dans ce cas, n’abandonnez pas le ranker, gardez 4-8 rollouts pendant l’inférence :
4.1 Verifier-override calibré (pivot la moins coûteuse, à essayer même avant
de pivoter pour de vrai)
Inspirée de la littérature LLM test-time compute (Snell et al. 2024 Scaling LLM test-time
compute, Setlur et al. 2025) qui montre que les méthodes verifier-based battent
strictement verifier-free quand la base policy est hétérogène, arXiv à condition que le
verifier soit calibré. Traduit pour vous :
Proposer : Step2 (rapide, stable)
Candidats : top-K legaux selon Step2
Verifier : ranker Step3 (instable mais utile en relatif)
Override : choisir argmax_ranker SUR les K candidats, ssi
ranker_top1_confidence > τ (après temperature scaling !)
ET ensemble_disagreement < δ
sinon : jouer Step2.
Escalade : si ensemble incertain ET enjeu fort → 4-8 rollouts CRN sur top-2.
Cette structure transforme votre ranker bruité en signal de gating plutôt qu’en politique.
Avant d’utiliser quoi que ce soit comme verifier, faites passer le ranker par temperature
scaling sur un set de validation (Guo et al., ICML 2017) — c’est 30 lignes de code, gratuit, et
indispensable pour que le seuil τ ait du sens. Métrique cible : ECE post-scaling < 2%, OE
(Overconfidence Error) près de zéro.
4.2 MO-IS-MCTS avec votre ranker comme prior PUCT — la pivot recherche
Si la pivot light verifier-override ne suffit pas, MO-IS-MCTS de
Cowling/Powley/Whitehouse 2012 White Rose Research Online University of York est
exactement la version « propre » de votre teacher rollout-guided. Différences essentielles :
Statistiques agrégées par infoset, pas par état → résout la « strategy fusion » qui
plombe PIMC pur (Long, Sturtevant, Buro 2010 Semantic Scholar — Love Letter, avec ses
Priest/Baron/King révélateurs, est précisément le cas où PIMC souffre).
PUCT : Q + c·P·√N/(1+n) avec P fourni par votre ranker (= prior policy), V_leaf fourni par
votre tête value Step2. C’est AlphaZero adapté à l’info imparfaite.
Déterminisations échantillonnées depuis votre belief net, pas uniformément. C’est le
bon usage de la tête belief.
EPIMC depth-2/3 (Arjonilla & Cazenave, 2024, arXiv 2408.02380) : différer l’évaluateur
perfect-info de quelques plies. Simple Science arXiv Très bon ROI engineering pour ~1
jour.
Coût de mise en œuvre : 1-2 jours sur votre code existant. Pas de ré-entraînement
nécessaire. Vous obtenez immédiatement la réponse à la question « la skill du teacher est-
elle search-bound ou prior-bound ? ».
4.3 Pourquoi pas full Deep CFR / SD-CFR / ReBeL / PoG
À 4 joueurs, CFR n’a aucune garantie de convergence vers Nash (Abou Risk &
Szafron 2010, Gibson 2014). Pluribus à 6 joueurs (Brown & Sandholm 2019, Science)
PubMed Carnegie Mellon University traite explicitement son blueprint MCCFR comme
heuristique, et fait toute la finesse en subgame depth-limited au play-time.
ReBeL et Player of Games sont des projets industriels arXiv NeurIPS (3-6 semaines
minimum d’engineering pour un dev seul ; pas de référence libre comparable à
OpenSpiel pour PoG).
Le benchmark Rudolph-Sokota 2026 (ICLR, Reevaluating Policy Gradient Methods for
Imperfect-Information Games, 7000 runs, 350 hyperparam configs) trouve que NFSP,
PSRO, ESCHER, R-NaD échouent à battre PPO/PPG/MMD arXiv sur les benchmarks
d’exploitabilité standard. Le coût engineering CFR ne se justifie pas tant qu’un
PG/distillation propre n’a pas été tentée.
SD-CFR (Steinberger 2019) dans OpenSpiel reste l’option CFR la plus raisonnable si
vous y tenez : ~4-7 jours de dev (il faut écrire Love Letter en OpenSpiel d’abord), 1-3
jours CPU. Mais c’est votre option de rang 5, pas de rang 1.
4.4 La pivot conceptuellement la plus juste : piKL
Si vous voulez vraiment passer à une formulation regret-minimization mais sans réécrire le
pipeline, la cible est piKL (Jacob, Wu, Farina, Lerer, Hu, Bakhtin, Andreas, Brown — NeurIPS
2022, le cœur de Cicero). Idée : régularisation KL vers une politique d’ancrage (votre Step2
distillé) tout en faisant du regret-minimization vers une best response. C’est
conceptuellement exactement le pattern « heuristic anchor → improve » que vous avez déjà
construit, formalisé en regret minimization régularisée. Quelques centaines de lignes, code
partiel public dans le repo Cicero. Si vous deviez pivoter une seule fois, ce serait là, pas
vers Deep CFR.
Variante encore plus légère : RCFR-lite (Waugh et al. 2015 + votre setup actuel). Remplacez
le softmax du policy head par regret-matching π(a|s) ∝ max(0, A(s,a)) (avec fallback
uniforme si tous A ≤ 0). C’est une ligne de code et vous convertit la distillation locale en
quelque chose qui minimise le regret par infoset — addresse la strategy-fusion au niveau
policy update gratuitement.
5. Plan d’expériences ordonné — « la plus petite étape qui
pourrait marcher »
Ordre strict, chaque étape débloque la suivante. Comptez ~1 semaine totale.
# Action Coût
eng
Coût
compute
Gain
attendu
À garder
ssi
1 CRN sharing + paired-t
filtrage
1
jour
0 (re-
collect)
5-20× var↓,
paires
utilisables
ρ_paired
> 0.5
2 Dueling head + pairwise
margin-conf
0.5
jour
1-2h
training
calibration
↓, oracle-
state ↑
ECE
drop,
oracle-
acc ↑
3 Temperature scaling +
verifier-override 2h 0
tail risk ↓,
robustesse
ECE post
< 2%,
override
gain ≥
teacher
gain
4 Diagnostics : oracle-state +
search-budget sweep
0.5
jour
1 jour
CPU
décide rang
5 vs 6
—
5a (si
search-
bound)
MO-IS-MCTS + PUCT prior
+ belief-determinisations +
EPIMC d=2
2
jours
0 gros gain
robuste
—
5b (si
prior-
bound)
DAgger soft / ThriftyDAgger
3 itérations
2-3
jours 1-2 jours +5-15 pts
win-rate
self-
induced
val ≈ iid
val
6
(option)
piKL anchored regret-
minimization ou RCFR-lite
1-3
jours
0.5-1
jour
strategy-
fusion ↓
si 5
plafonne
7
(option
lourde)
SD-CFR dans OpenSpiel 4-7
jours 1-3 jours blueprint
indépendant
si tout le
reste
plafonne
La séquence 1→2→3→4 est non-négociable et tient en 2-3 jours sur Mac Pro. Elle
adresse les trois failles diagnostiquées (bruit MC, mauvais objet appris, mismatch verifier).
Elle fournit aussi le diagnostic search-bound vs prior-bound qui guide la suite. Ne lancez 5
ou 6 qu’après avoir vu où plafonne 1-4.
6. Diagnostics pour décider « persister vs pivoter »
Cinq tests à 1-2h chacun. Vous les faites avant de décider 5a vs 5b vs 6.
D1 — Train/val gap auto-induit : split le val en (a) i.i.d. et (b) auto-induit (états visités par le
student courant en self-play). Si val_
iid ≈ ok mais val
_
self ≫, c’est covariate shift → DAgger.
D2 — Seed-agreement : entraînez K=3-5 seeds sur les mêmes labels. Mesurez top-1
disagreement sur held-out.
Disagreement >20% + train-acc OK → label noise domine → augmentez rollouts, soft
labels, CRN.
Disagreement <5% + perf plafonne → capacité ou objet appris → dueling, ensemble,
ou pivot search.
D3 — Oracle-state accuracy (le diagnostic clé). Définir O = { s : argmax Q_teacher(s) ≠
argmax Q_Step2(s) ET marge teacher ≥ ε }. Mesurer top-1 accuracy du ranker uniquement
sur O. Win-rate moyen est insensible à O (fraction petite des états). Oracle-state acc plat
≈ aucune des gains stratégiques absorbée → distillation échoue là où ça compte. Signal de
pivot 5a.
D4 — Search-budget sweep : évaluez votre teacher à B ∈ {1, 4, 16, 64, 256} rollouts/action
contre Step2. Trace la courbe.
Sature à B≤4 → skill local, ranker peut compresser → persister sur distillation (rang 5b).
Continue à monter à B=64 → skill = recherche → 5a obligatoire, n’essayez pas de la
compresser.
D5 — Calibration & ECE : reliability diagram ranker ; appliquer temperature scaling ;
mesurer ECE pré et post. Brier score sur l’action choisie. Si post-temp ECE < 2% mais
comportement vs Step2 stable, votre verifier-override (#3) est viable. Sinon, ensemble
nécessaire.
Pivot trigger composé : si D3 plat ET D4 toujours steep à B=64 ET D2 disagreement
modéré, arrêtez de chercher à compresser la recherche — la skill est la recherche.
Verifier-override (#3) ou MO-IS-MCTS (#5a) sont les bonnes réponses.
7. Évaluer un agent destiné à 1-2 parties contre un humain
Votre objectif final n’est pas un win-rate moyen sur 1000 sims. AIVAT (Burch, Schmid,
Moravčík, Bowling AAAI 2018) est explicite : « des matchs de plusieurs jours homme-
machine en HUNL ne donnent toujours pas de conclusions statistiquement significatives ».
À 1-2 parties, ce qui compte est la queue de distribution et la perception.
Suite d’éval recommandée (~quelques heures CPU) :
1. Set dupliqué-permuté : 50 deals × 24 permutations de sièges = 1200 parties. Calculez
moyenne ET 10ᵉ percentile de score margin. C’est l’analogue Pluribus/Cicero pour
réduire la variance de seat draw.
2. Approximate exploitability : entraînez un petit DQN/PPO uniquement contre votre
candidate + 3 Step2 (Timbers et al. IJCAI 2022). Si l’exploiter atteint ≫0, votre
candidate a un trou — un humain peut le trouver en 2 parties.
3. 4. 5. 6. 7. 8. Oracle-state subset accuracy (D3 ci-dessus) — la seule métrique qui mesure
directement si votre distillation a absorbé les gains stratégiques.
Calibration : ECE post-temperature-scaling, OE (Overconfidence Error) — un humain
qui voit un blunder en game 1 vous range pour la game 2.
Blunder rate : fraction de parties contenant ≥1 action avec marge teacher > τ depuis la
max-action. Plus parlant qu’un win-rate.
Behavioral panel : action entropy par phase, EAV (Expected Action Variation entre
seeds), Wasserstein behavioral distance vs Step2. Si distance ≈ 0 mais win-rate >50%,
vous exploitez Step2, vous ne jouez pas mieux à Love Letter — ça ne survivra pas à
l’humain.
TrueSkill league (Herbrich-Minka-Graepel NIPS 2006) avec population gelée {random,
heuristic, BC, Step2, teacher@B=1/4/16/64, candidate}. ~200 parties par paire, µ±2σ.
Stop quand σ < 0.5×Δµ utile.
Pilote humain : 6-10 parties + débrief structuré (Cicero a fait 40 parties en eval finale).
Questions : « le bot a-t-il fait quelque chose d’inattendu / stupide / inquiétant ? ». C’est
qualitatif mais informationnellement dense.
8. Pièges concrets à éviter
Strategy fusion dans le teacher : si vos rollouts utilisent l’état réel (et pas une
déterminisation depuis le belief), le teacher a accès à des infos que le ranker n’aura pas
→ vous distillez de la clairvoyance. Vérifiez que vos déterminisations sont samplées du
belief, pas de la ground truth. C’est le piège #1 de PIMC documenté par Long et al. 2010.
Override sans gating de confiance → label flips dans le dataset (action A override B en
s, B override A en s’ quasi-identique). Symptôme : pair-accuracy validation plafonne
~60%. Fix : §2.3.
Ré-entraînement from scratch à chaque DAgger iter → instabilité. Toujours fine-tune
avec lr réduit.
β=1 mixture en DAgger trop longtemps → retour vers BC. β=0 dès iter 1 (Ross 2011
standard).
Ensemble sans diversité : K seeds qui convergent au même minimum n’apportent rien
(Abe et al. 2022). Vérifiez par KL pairwise sur held-out ; si <0.05, votre ensemble est
fictif.
Évaluer uniquement vs Step2 : Step2 peut avoir des trous spécifiques. Évaluez aussi
vs teacher@small-budget et teacher@large-budget pour mesurer le gap réel de
distillation.
Goodhart sur le rollout signal : si vous fittez trop fort le ranker au teacher@N=16, il
peut mémoriser les patterns de bruit MC à N=16. Validez sur teacher@N=256 (un sous-
set).
Pondération par phase de partie sans replay stratifié : les phases de fin (rares mais
critiques) sont écrasées. Toujours stratifier par phase dans le sampler.
Signal positif fort que vous êtes sur la bonne voie : oracle-state accuracy monte, seed-
agreement reste élevé, ECE post-temp baisse, le gap teacher@B=large vs student diminue
avec données. Signal négatif fort : oracle-state accuracy plat malgré dataset×2, seed-
disagreement >20%, ECE non récupérable par temp-scaling — c’est le moment de §4.
9. Travaux 2023-2026 utiles
Update-Equivalence Search (Sokota et al., ICLR 2024, arXiv 2304.13138) : recherche
en jeu imparfait sans construire de PBS. Bat SPARTA sur Hanabi avec 100× moins de
compute. Le plus pertinent pour Love Letter (private info distribué). Research-grade à
implémenter, mais idée applicable.
piKL (Jacob et al., NeurIPS 2022) — déjà discuté, le pivot conceptuelle correcte.
Rudolph-Sokota et al., ICLR 2026 — « Reevaluating Policy Gradient Methods for
Imperfect-Information Games ». PG bien tuné = CFR profond sur exploitabilité. Justifie
de ne pas pivoter vers CFR.
ESCHER (McAleer et al., ICLR 2023) : Deep CFR sans importance sampling, bat
DREAM/NFSP. Si vous voulez vraiment Deep CFR.
AlphaZe** (Czech et al., Frontiers AI 2023) : MuZero-shape pour info imparfaite avec
PUCT prior + belief input. Le template moderne pour 5a.
DeepRole (Serrino et al., NeurIPS 2019) sur Avalon — IS-MCTS + belief inference, le
plus proche de votre cas.
Sur Mac Pro : MLX d’Apple ou PyTorch MPS sont matures pour les MLPs/transformers
<10M params (votre cas). Le bottleneck reste le simulateur de rollouts ; investir 2 jours dans
un sim Numba/Cython/Rust de Love Letter est le meilleur multiplicateur de toute la chaîne.
10. Conclusion exécutive
Trois choses à retenir.
Le problème n’est pas que la distillation locale soit fondamentalement limitée pour
Love Letter — c’est que vous avez fait fonctionner les deux étapes faciles (BC, retarget
belief) et vous attaquez maintenant l’étape où la littérature est explicite sur les
techniques nécessaires (variance reduction, advantage formulation, label calibration)
sans encore les avoir appliquées. La séquence CRN → dueling+pairwise margin-conf →
filtrage paired-t → temperature scaling+verifier-override est la « plus petite étape qui
pourrait marcher » et tient en 2-3 jours. Faites-la.
Le diagnostic search-bound vs prior-bound est l’unique question stratégique qui
reste. Le search-budget sweep de votre teacher la résout en une journée CPU. Si search-
bound : MO-IS-MCTS avec votre ranker en PUCT prior et belief-determinizations, plus
EPIMC depth 2-3. Pas CFR, pas ReBeL, pas Player of Games — ce sont des chantiers
industriels et 4 joueurs leur retire leurs garanties théoriques de toute façon.
Pour l’éval vs humain à 1-2 parties, abandonnez le win-rate moyen comme métrique
primaire. Le triplet (score-margin distribution, blunder rate, exploitabilité approximée) + un
pilote humain de 6-10 parties est plus informatif que 10k sims uniformes. Calibrez votre
verifier (temperature scaling, ECE<2%) avant d’autoriser tout override — un humain qui voit
un blunder en game 1 ne reviendra pas en game 2.
L’asymétrie engineering-vs-gain est très favorable : les corrections principales sont small
code-level changes (10-100 LoC chacune) qui composent multiplicativement. La seule
chose à ne pas faire avant de les avoir tentées est de basculer vers une réécriture CFR ou
un projet ReBeL. Vous n’avez pas encore touché au plafond de la distillation locale ; vous
avez touché au plafond d’une distillation locale naïve.
# Fiche modèle — classical-emotion 2.9.0

Paquet publié. L'alias de production reste classical-emotion 2.6.0. Les checkpoints 2.6.0, 2.7.0 et 2.8.0 ne sont pas modifiés. Le chargement se fait par `SessionNeutralModel.from_release_dir`, pas par le prédicteur de production.

## Données

Même jeu étiqueté que 2.7.0 et 2.8.0. P004, P008 et P009 exclus de l'ajustement. Coupe-bande 60 Hz (Q = 30) puis 20–110 Hz. RMS sur les échantillons valides. Fenêtres sous 80 % de couverture ou au-dessus de log(RMS) 10,5 rejetées. Aucun retrait pour étiquette douteuse.

## Normalisation

Neutre de session par canal, N = w · p20 + (1 − w) · prior, w = n / (n + 60), prior global de la ronde 3. Feature d'amplitude : log(RMS) − N. Waveform length des canaux EMG, même règle, prior estimé sur le labo inclus. Fréquence médiane, asymétrie zygomatique, corrélation antagoniste et NM-A conservés, sans remise à l'échelle de session. Pas de direction adoucie.

Le même calcul sert à l'entraînement, à l'évaluation et à l'inférence. En temps réel, N part du prior, est mis à jour toutes les 10 s sur le passé valide, et varie d'au plus 0,05 log par mise à jour.

## Couche de décision

Régression logistique (C = 1). Température ajustée hors de la session test. Seuil par émotion au maximum du F0,5. Sous le seuil : NEUTRAL. Écrêtage du RMS au p99,5 étiqueté de la ronde 1, avec indicateur des canaux écrêtés. Pas d'abstention. Aucune classe n'est éteinte, y compris FEAR.

## Classes émises

NEUTRAL, HAPPINESS, ANGER, DISGUST, FEAR, SURPRISE, SADNESS, CONTEMPT (CONTEMPT_LEFT et CONTEMPT_RIGHT sont fusionnés à la sortie).

## Limites connues

Le LOSO imbriqué sur les 21 sujets inclus est 0,476 [0,423, 0,529]. Sur Malefycia le candidat est moins souvent NEUTRAL que 2.6.0. Le p20 de labo peut s'écarter de la médiane des fenêtres NEUTRAL. Le lissage a posteriori de production n'est pas dans ce paquet. Les classes sous le seuil de précision 0,50 ou de rappel 0,10 restent émises.

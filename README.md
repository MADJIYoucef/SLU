# Modèle joint Slot filling & Intent detection.
Ce projet est une implémentation d'un modèle joint pour Slot filling et Intent detection.<br>
Pour lancer l'étape entrainement executer la commande suivante :<br>
<b>python -W ignore train.py --m <num de modèle></b><br>
Pour lancer l'étape test executer la commande suivante :<br>
<b>python -W ignore test.py --m <num de modèle></b><br>

avec num modèle un nombre entre 1 et 4

<ul>
  <li> 1 : 2 Couches GRU</li>
  <li> 2 : BiLSTM</li>
  <li> 3 : Encodeur décodeur GRU</li>
  <li> 4 : Conv encodeur-décodeur</li>
</ul>

Pour plus de détails voir le note book : "spoken_language_understanding.ipynb" ou il ya une implémentation des deux taches séparements

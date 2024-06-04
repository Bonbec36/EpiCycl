<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dessin de Formes avec des Cercles Utilisant des Transformées de Fourier</title>
</head>
<body>
    <h1>Dessin de Formes avec des Cercles Utilisant des Transformées de Fourier</h1>
    <p>Ce projet utilise des transformées de Fourier pour dessiner des formes complexes en utilisant des cercles. L'objectif est de décomposer une forme en une série de cercles qui tournent autour de points centraux, chaque cercle représentant une composante fréquentielle de la forme.</p>
    
    <h2>Table des matières</h2>
    <ul>
        <li><a href="#introduction">Introduction</a></li>
        <li><a href="#pré-requis">Pré-requis</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#usage">Usage</a></li>
        <li><a href="#exemples">Exemples</a></li>
        <li><a href="#contribuer">Contribuer</a></li>
        <li><a href="#licence">Licence</a></li>
    </ul>
    
    <h2 id="introduction">Introduction</h2>
    <p>Les transformées de Fourier sont un outil mathématique puissant permettant de décomposer une fonction en une somme de sinusoïdes. Dans ce projet, nous utilisons cette technique pour représenter des formes complexes par des cercles en mouvement. Chaque cercle correspond à une composante fréquentielle de la forme d'origine.</p>
    
    <h2 id="pré-requis">Pré-requis</h2>
    <p>Avant de commencer, assurez-vous d'avoir les éléments suivants installés sur votre machine :</p>
    <ul>
        <li>Python 3.x</li>
        <li>Bibliothèques Python suivantes :
            <ul>
                <li><code>numpy</code></li>
                <li><code>matplotlib</code></li>
            </ul>
        </li>
    </ul>
    
    <h2 id="installation">Installation</h2>
    <p>Clonez ce dépôt sur votre machine locale :</p>
    <pre><code>git clone https://github.com/votre-utilisateur/fourier-draw.git
cd fourier-draw
</code></pre>
    <p>Installez les dépendances requises :</p>
    <pre><code>pip install numpy matplotlib
</code></pre>
    
    <h2 id="usage">Usage</h2>
    <p>Pour dessiner une forme avec des cercles utilisant des transformées de Fourier, exécutez le script <code>fourier_draw.py</code>. Vous pouvez spécifier une forme prédéfinie ou fournir vos propres coordonnées.</p>
    <p>Exemple de commande :</p>
    <pre><code>python fourier_draw.py --shape star
</code></pre>
    <p>Options disponibles :</p>
    <ul>
        <li><code>--shape</code> : spécifie la forme à dessiner. Valeurs possibles : <code>star</code>, <code>square</code>, <code>custom</code>.</li>
        <li><code>--file</code> : chemin vers un fichier CSV contenant les coordonnées X, Y pour les formes personnalisées.</li>
    </ul>
    
    <h2 id="exemples">Exemples</h2>
    <h3>Étoile</h3>
    <img src="examples/star.png" alt="Étoile">
    <h3>Carré</h3>
    <img src="examples/square.png" alt="Carré">
    <h3>Forme Personnalisée</h3>
    <p>Pour dessiner une forme personnalisée, fournissez un fichier CSV avec les colonnes <code>X</code> et <code>Y</code> :</p>
    <pre><code>python fourier_draw.py --shape custom --file path/to/your/coordinates.csv
</code></pre>
    <h3>Animation</h3>
    <p>Vous pouvez également générer une animation montrant les cercles en mouvement recréant la forme :</p>
    <pre><code>python fourier_draw.py --shape star --animate
</code></pre>
    
    <h2 id="contribuer">Contribuer</h2>
    <p>Les contributions sont les bienvenues ! Veuillez soumettre un <code>pull request</code> ou ouvrir une <code>issue</code> pour discuter des changements que vous souhaitez apporter.</p>
    
    <h2 id="licence">Licence</h2>
    <p>Ce projet est sous licence MIT. Voir le fichier <a href="LICENSE">LICENSE</a> pour plus de détails.</p>
</body>
</html>

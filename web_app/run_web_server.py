# run_web_server.py (à la racine du projet)
from web_app.main import app # ou web_app.app si votre fichier s'appelle app.py

if __name__ == '__main__':
    # Le CORE_OUTPUT_DIR est défini dans web_app/main.py.
    # On s'assure qu'il est créé si ce n'est pas déjà fait.
    # Note: le serveur Flask ne doit pas être utilisé en production tel quel pour servir des fichiers statiques de manière performante.
    # Pour la prod, on utiliserait Nginx/Apache devant.
    # Mais pour le dev, c'est suffisant.

    print("Lancement du serveur de développement Flask...")
    print("Accédez à http://127.0.0.1:5001/ dans votre navigateur.")
    app.run(debug=True, host='0.0.0.0', port=5001)
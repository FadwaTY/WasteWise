# wastewise/llm_reco.py
import os
from google import genai

# --- CONFIGURATION GLOBALE ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "La variable d'environnement GEMINI_API_KEY est manquante !"
    )

# Initialise le client
client = genai.Client(api_key=API_KEY)

# Nom du modèle recommandé
MODEL_NAME = "gemini-2.5-flash"

# --- FONCTION DE RECOMMANDATION ---
def get_recycling_advice(class_name: str) -> str:
    """
    Génère une recommandation intelligente pour un type de déchet détecté.
    """
    prompt = f"""
    Tu es un assistant intelligent spécialisé en recyclage.
    Le déchet détecté est : {class_name}.

    Donne une recommandation simple, précise et actionable :
    - Comment recycler ce type de déchet ?
    - Quel impact environnemental cela évite ?
    - Une idée d’upcycling possible.
    Répond en 3 phrases maximum.
    """

    try:
        # Appel au modèle Gemini via la nouvelle API
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text  # Le texte généré

    except Exception as e:
        # Gestion des erreurs (clé invalide, modèle indisponible, problème réseau, etc.)
        print(f"❌ Erreur lors de l'appel LLM: {e}")
        return "Erreur de service de recommandation. Le modèle pourrait être indisponible ou la clé invalide."

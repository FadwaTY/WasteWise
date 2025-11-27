import os
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Donne un conseil simple pour recycler du plastique."
    )
    print("✅ Réponse reçue :")
    print(response.text)

except Exception as e:
    print("❌ Erreur lors de l'appel à l'API Gemini :")
    print(e)

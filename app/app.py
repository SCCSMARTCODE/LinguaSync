from flask import Flask, render_template, request, jsonify
from googletrans import Translator

app = Flask(__name__)
translator = Translator()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.get_json()
        english_text = data.get("text", "")

        if not english_text:
            return jsonify({"error": "No text provided"}), 400

        # Perform translation
        print("Hi")
        translated = translator.translate(english_text, src="en", dest="fr")
        print("Hi1")
        print(english_text, translated)
        return jsonify({"translation": translated.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

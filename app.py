from flask import Flask, render_template, request, redirect, session, jsonify
import pickle
import sqlite3
import re
import numpy as np

app = Flask(__name__)
app.secret_key = "secret123"

# ---------------- LOAD MODELS ----------------
try:
    model = pickle.load(open('models/nlp_model.sav', 'rb'))
    vectorizer = pickle.load(open('models/Vectorizer', 'rb'))
except Exception as e:
    print("❌ Error loading model:", e)
    model = None
    vectorizer = None

# ---------------- LABELS ----------------
labels = ['business', 'sport', 'politics', 'tech', 'entertainment']

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect('users.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            password TEXT
        )
    ''')
    conn.close()

init_db()

# ---------------- KEYWORDS ----------------
keywords = {
    'business': {'market':2,'company':2,'stock':3,'bank':2,'profit':2,'economy':3,'finance':2,'investment':2},
    'sport': {'match':2,'team':2,'player':2,'score':2,'cricket':3,'football':3,'league':2,'goal':2},
    'politics': {'government':3,'election':3,'minister':2,'policy':2,'law':2,'vote':2},
    'tech': {'software':2,'technology':3,'ai':3,'computer':2,'data':2,'internet':2},
    'entertainment': {'movie':2,'music':2,'film':2,'actor':2,'show':2,'celebrity':2},

    # NEW
    'education': {'school':2,'college':2,'university':3,'student':2,'exam':2,'teacher':2},
    'health': {'hospital':3,'doctor':3,'health':2,'disease':2,'medicine':2,'vaccine':3},
    'science': {'research':3,'experiment':3,'scientist':3,'physics':2,'chemistry':2,'biology':2}
}

# ---------------- SAMPLE TEXTS ----------------
sample_texts = {
    "business": [
        "Stock markets saw a sharp rise in tech shares today.",
        "The company reported increased quarterly profits.",
        "Banks are investing heavily in digital transformation.",
        "The economy is showing signs of recovery.",
        "New startup funding reached record levels."
    ],
    "sport": [
        "The team won the match with a last-minute goal.",
        "Cricket tournament finals will be held tomorrow.",
        "The player scored a century in the game.",
        "Football league matches are heating up.",
        "The coach praised the team's performance."
    ],
    "politics": [
        "The government announced a new policy reform.",
        "Elections will be held next month.",
        "The minister addressed the parliament today.",
        "New law proposals are under discussion.",
        "Political campaigns are increasing."
    ],
    "tech": [
        "AI is transforming modern technology rapidly.",
        "Cybersecurity is a major concern today.",
        "Tech companies invest in cloud computing.",
        "Data science demand is increasing.",
        "New software innovations are emerging."
    ],
    "education": [
        "Students are preparing for final exams.",
        "Universities are offering new courses.",
        "Online learning is growing rapidly.",
        "Teachers are adapting to digital classrooms.",
        "Education reforms are introduced."
    ],
    "health": [
        "Doctors are monitoring patient recovery.",
        "Vaccination programs are expanding.",
        "Hospitals are improving services.",
        "New treatments are being researched.",
        "Health awareness campaigns are rising."
    ],
    "science": [
        "Scientists discovered a new particle.",
        "Space research is expanding rapidly.",
        "Experiments showed promising results.",
        "Biology studies are advancing.",
        "New discoveries are transforming science."
    ],
    "entertainment": [
    "The movie received great reviews from critics.",
    "The actor delivered an outstanding performance.",
    "Music concerts are gaining huge popularity.",
    "The television show attracted a large audience.",
    "Celebrities attended the grand award ceremony."
]
}

# ---------------- ROUTES ----------------

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user = request.form.get('username')
        pwd = request.form.get('password')

        if not user or not pwd:
            return "⚠️ Fill all fields"

        conn = sqlite3.connect('users.db')
        conn.execute("INSERT INTO users (username, password) VALUES (?,?)", (user, pwd))
        conn.commit()
        conn.close()

        return redirect('/login')

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('username')
        pwd = request.form.get('password')

        conn = sqlite3.connect('users.db')
        cursor = conn.execute("SELECT * FROM users WHERE username=? AND password=?", (user, pwd))
        data = cursor.fetchone()
        conn.close()

        if data:
            session['user'] = user
            return redirect('/predict')
        else:
            return "❌ Invalid Credentials"

    return render_template('login.html')


# ---------------- EXAMPLES API ----------------
# ---------------- EXAMPLES API ----------------
@app.route('/examples/<category>')
def examples(category):
    return jsonify(sample_texts.get(category.lower(), []))


# ---------------- PREDICT ----------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect('/login')

    prediction = None
    confidence = 0
    text = ""

    if request.method == 'POST':
        text = request.form.get('text', '').strip()

        if text == "":
            return render_template('predict.html',
                                   prediction=None,
                                   confidence=0,
                                   text=text)

        cleaned_text = clean_text(text)

        # ML prediction
        if model and vectorizer:
            try:
                vec = vectorizer.transform([cleaned_text])
                probs = model.predict_proba(vec)[0]
                idx = int(np.argmax(probs))

                prediction = labels[idx]
                confidence = round(float(probs[idx]) * 100, 2)

            except:
                prediction = None
                confidence = 0

        # Keyword scoring
        text_lower = text.lower()
        scores = {}

        for category, words in keywords.items():
            score = 0
            for word, weight in words.items():
                score += weight * text_lower.count(word)
            scores[category] = score

        best_keyword = max(scores, key=scores.get)

        if scores[best_keyword] >= 5:
            prediction = best_keyword
            confidence = max(confidence, 85)

        elif confidence < 60 and scores[best_keyword] > 0:
            prediction = best_keyword
            confidence = max(confidence, 70)

    return render_template('predict.html',
                           prediction=prediction,
                           confidence=confidence,
                           text=text)


# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')


# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)

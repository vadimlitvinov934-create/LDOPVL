import os
from flask import Flask, render_template, request, jsonify
from rag_engine import RAG

app = Flask(__name__)

KEY = 'AIzaSyDkNP4CgpMfh2WFBOTLHtdVTq5ok-grNRs'

print(">>> Запуск LDO Brain...")
brain = RAG(KEY)

DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Создал папку {DATA_DIR}, кидай туда файлы.")
else:
    brain.load_data(DATA_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    try:
        msg = request.json.get('message')
        if not msg: return jsonify({'error': 'Пусто'}), 400

        res = brain.ask(msg)
        
        return jsonify({
            'response': res['ans'],
            'sources': res['refs']
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
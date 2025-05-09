import pandas as pd
import ollama
import random
import time
import csv
import io
import json
from flask import Flask, request, jsonify, render_template, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
df = None


def load_data():
    global df
    try:
        df = pd.read_csv("plusbig5_result.csv")
        print("CSV 파일이 성공적으로 로드되었습니다.")
    except Exception as e:
        print(f"CSV 파일 로드 중 오류 발생: {str(e)}")
        return False
    return True


def analyze_input(user_input, df):
    # 사용자 입력을 키워드로 변환
    keywords = user_input.split()

    # Ollama를 사용하여 감정 분석
    emotion_prompt = f"""Analyze the emotional content of this text and determine the dominant emotion:
    Text: {user_input}
    Choose one dominant emotion from: fear, sadness, anger, joy, love, surprise.
    Return only the emotion word."""

    emotion_response = ollama.generate(
        model='llama3',
        prompt=emotion_prompt,
        options={
            'temperature': 0.3,
            'top_p': 0.9,
            'num_predict': 20,
        }
    )

    user_emotion = emotion_response['response'].strip().lower()

    # 유사한 데이터 찾기
    vectorizer = TfidfVectorizer()
    all_keywords = df['keywords'].fillna('').tolist()
    all_keywords.append(' '.join(keywords))

    tfidf_matrix = vectorizer.fit_transform(all_keywords)
    similarity_scores = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]

    # 감정이 일치하는 행만 필터링
    emotion_matches = df['dominant_emotion'] == user_emotion
    if emotion_matches.any():
        emotion_indices = emotion_matches[emotion_matches].index
        best_match_idx = emotion_indices[similarity_scores[emotion_indices].argmax()]
    else:
        best_match_idx = similarity_scores.argmax()

    similar_data = df.iloc[best_match_idx]

    return {
        'keywords': keywords,
        'emotion': user_emotion,
        'similar_data': similar_data
    }


def generate_dialogue(analysis_result):
    keywords = analysis_result['keywords']
    emotion = analysis_result['emotion']
    similar_data = analysis_result['similar_data']

    # 프롬프트 생성
    if emotion == 'sadness' and pd.notna(similar_data['interpreted_emotion_flavor']):
        emotion_flavor = similar_data['interpreted_emotion_flavor']
        prompt = f"""Create a dialogue that reflects the emotion '{emotion}' with the specific flavor of '{emotion_flavor}' and incorporates these keywords: {', '.join(keywords)}.
        The dialogue should be natural and emotionally appropriate, capturing the nuanced sadness described.
        Return only the dialogue line without any additional text or explanation.
        Dialogue:"""
    else:
        prompt = f"""Create a dialogue that reflects the emotion '{emotion}' and incorporates these keywords: {', '.join(keywords)}.
        The dialogue should be natural and emotionally appropriate.
        Return only the dialogue line without any additional text or explanation.
        Dialogue:"""

    # Ollama를 사용하여 대화 생성
    response = ollama.generate(
        model='llama3',
        prompt=prompt,
        options={
            'temperature': 0.7,
            'top_p': 0.9,
            'num_predict': 100,
            'repeat_penalty': 1.1,
            'top_k': 40,
            'seed': random.randint(1, 10000)
        }
    )

    return response['response'].strip()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate_dialogue_endpoint():
    if df is None:
        return jsonify({'error': 'Data not loaded'}), 500

    data = request.get_json()
    if not data or 'input' not in data:
        return jsonify({'error': 'No input provided'}), 400

    user_input = data['input']

    try:
        print(f"\n입력 분석 중...")
        analysis_result = analyze_input(user_input, df)

        print(f"\n분석 결과:")
        print(f"감정: {analysis_result['emotion']}")
        if analysis_result['emotion'] == 'sadness' and pd.notna(
                analysis_result['similar_data']['interpreted_emotion_flavor']):
            print(f"감정 세부: {analysis_result['similar_data']['interpreted_emotion_flavor']}")

        print("\n대사 생성 중...")
        dialogue = generate_dialogue(analysis_result)

        return jsonify({
            'dialogue': dialogue,
            'emotion': analysis_result['emotion'],
            'emotion_flavor': analysis_result['similar_data']['interpreted_emotion_flavor'] if analysis_result[
                                                                                                   'emotion'] == 'sadness' else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download-history', methods=['GET'])
def download_history():
    try:
        # 클라이언트로부터 전송된 히스토리 데이터 받기
        history_data = request.args.get('data')
        if not history_data:
            return jsonify({'error': 'No history data provided'}), 400

        # JSON 데이터 파싱
        history = json.loads(history_data)

        # CSV 데이터 생성
        output = io.StringIO()
        writer = csv.writer(output)

        # 헤더 작성
        writer.writerow(['입력', '생성된 대사', '감정', '감정 세부', '생성 시간'])

        # 데이터 작성
        for item in history:
            writer.writerow([
                item['input'],
                item['dialogue'],
                item['emotion'],
                item.get('emotion_flavor', ''),
                item['timestamp']
            ])

        # CSV 파일 생성
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'dialogue_history_{time.strftime("%Y%m%d")}.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    if load_data():
        app.run(host='0.0.0.0', port=5000, debug=True)

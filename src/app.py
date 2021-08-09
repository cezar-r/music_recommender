from flask import Flask, render_template, request
from artist_classifier import Classifier
from datetime import datetime

app = Flask(__name__)

class LyricPredictor:

	def __init__(self):
		self.load_model()

	def predict(self, lyric):
		artist_predictions = self.model.predict_one(lyric)
		return artist_predictions

	def load_model(self):
		sg = Classifier()
		self.model = sg.load()


clf = LyricPredictor()

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
	lyric = str(request.form['lyric'])
	now = datetime.now()
	artists = clf.predict(lyric)
	print(datetime.now() - now)
	# print(artists)
	return render_template('index_pred.html', data = (artists, lyric))


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080, debug=True)
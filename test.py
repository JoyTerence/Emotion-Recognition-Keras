from keras.models import load_model
import displayImage

model = load_model('my_model.h5')
emotion = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
def evaluate(data, label):

    model.evaluate(x=data, y=label, batch_size=32, verbose=1)

def predict(data, label):

    emotion_list = model.predict(data)
    pred_list = [angry, disgust, fear, happy, sad, surprise, neutral] = [prob for lst in emotion_list for prob in lst]
    pred_dict = dict(zip(emotion, pred_list))
    sorted_by_value = sorted(pred_dict.items(), key=lambda kv: kv[1], reverse=True)
    print (sorted_by_value[0][0])

#img, data = webcam.fromImage(r'neutral1.jpg')

#displayImage.plotImagePixels(data, "sad")

#predict(img, "sad")

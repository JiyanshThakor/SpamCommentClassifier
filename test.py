import joblib as jb

model = jb.load("SpamPredicter.pkl")

prediction = model.predict(["Sub for free moni"])

if prediction[0] == 1:
    print("Spam")
else:
    print("Real")
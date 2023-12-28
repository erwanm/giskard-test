from transformers import pipeline

distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    top_k=None
)

print('EXPECTED pos = 0.9731044769287109 0.9760093688964844 0.9342429041862488')

# english
print(distilled_student_sentiment_classifier ("I love this movie and i would watch it again and again!"))



# malay
print(distilled_student_sentiment_classifier("Saya suka filem ini dan saya akan menontonnya lagi dan lagi!"))

# japanese
print(distilled_student_sentiment_classifier("私はこの映画が大好きで、何度も見ます！"))


from pytest import approx
from chameleon.models import HuggingFaceModel, Gpt2TrlProbe


def test_run_gpt2trl_probe_with_huggingface_model():
    model = HuggingFaceModel(
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )
    probe = Gpt2TrlProbe(TODO)
    input_sentence = "My grandmother's secret sauce is the best ever made!"
    epsilon = 1e-2
    input_sentence_sentiment = model.predict(input_sentence)
    target_sentence = probe.predict(model, input_sentence, epsilon)
    target_sentence_sentiment = model.predict(target_sentence)
    assert set(model.labels()) == set(["negative", "neutral", "positive"])
    for sentiment in model.labels():
        assert target_sentence_sentiment[sentiment] == approx(input_sentence_sentiment[sentiment], abs=epsilon)

from pytest import approx
from chameleon.models import HuggingFaceModel
from chameleon.probes import Gpt2TrlProbe


def test_run_gpt2trl_probe_with_huggingface_model():

    model = HuggingFaceModel(
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )
    assert set(model.labels()) == set(["negative", "neutral", "positive"])
    input_sentence = "My grandmother's secret sauce is the best ever made!"
    input_sentence_sentiment = model.predict(input_sentence)
    epsilon = 1e-3
    sample_size = 4096

    probe = Gpt2TrlProbe(model, input_sentence)
    probe_result = probe.run(epsilon=epsilon, sample_size=sample_size)
    print(f"Input sentence: {input_sentence}")
    print(f"Sentiment input sentence: {input_sentence_sentiment}")
    print(f"Output sentence: {probe_result.sentence}")
    print(f"Sentiment output sentence: {probe_result.scores}")
    for sentiment in model.labels():
        assert probe_result.scores[sentiment] == approx(input_sentence_sentiment[sentiment], abs=epsilon)


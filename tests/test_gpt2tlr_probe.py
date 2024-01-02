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

    probe = Gpt2TrlProbe(model, target)
    probe_result = probe.run(epsilon = 1e-2) # TODO
    target_sentence_sentiment = model.predict(probe_result.sentence)
    for sentiment in model.labels():
        assert probe_result.scores[sentiment] == approx(input_sentence_sentiment[sentiment], abs=epsilon)

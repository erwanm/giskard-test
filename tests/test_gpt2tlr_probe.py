from pytest import approx
from chameleon.models import HuggingFaceModel
from chameleon.probes import Gpt2TrlProbe
from chameleon.probes import softmax_dict


def test_run_gpt2trl_probe_with_huggingface_model():

    model = HuggingFaceModel(
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )
    assert set(model.labels()) == set(["negative", "neutral", "positive"])
    input_sentence = "My grandmother's secret sauce is the best ever made!"
    input_sentence_sentiment = model.predict(input_sentence)
    # TODO
    epsilon = 1e-3
    sample_size = 512

    probe = Gpt2TrlProbe(model, input_sentence)
    probe_result = probe.run(epsilon=epsilon, sample_size=sample_size)
    output_sentence_sentiment = model.predict(probe_result.sentence)
    print(f"Sentiment input sentence: {input_sentence_sentiment}")
    print(f"Sentiment output sentence: {output_sentence_sentiment}")
    scores = probe_result.scores
    for sentiment in model.labels():
        assert scores[sentiment] == approx(output_sentence_sentiment[sentiment], abs=epsilon)

# TODO
test_run_gpt2trl_probe_with_huggingface_model()

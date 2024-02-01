import matplotlib.pyplot as pyplot

from utils.video_summarization import upsample_sequence

def plot(summary, ground_truth_score, key):
    ground_truth_score = ground_truth_score.numpy()
    ground_truth_score = upsample_sequence(ground_truth_score)
    ground_truth_score = ground_truth_score[:len(summary)]

    for index in range(len(summary)):
        if summary[index] == 1:
            summary[index] = ground_truth_score[index]

    indexes = list(range(len(summary)))

    pyplot.subplot(2, 1, 1)
    pyplot.plot(indexes, ground_truth_score, color = 'silver')
    pyplot.fill_between(indexes, ground_truth_score, color = 'silver')
    pyplot.plot(indexes, summary, color = 'C0')
    pyplot.fill_between(indexes, summary, color = 'C0')
    pyplot.title(key)
    pyplot.show()
    return

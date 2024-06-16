import gensim.downloader as api
import spacy


class AnswerGenerator:
    def __init__(self, configs):
        self.glove_model = api.load(configs["models"]["glove"])
        self.spacy_model = spacy.load(configs["models"]["spacy"])
    
    def __filter_candidate_words(self, user_topic, candidate_words, max_len):
        candidate_words = [word for word in candidate_words if user_topic not in word]

        user_topic_lemma = self.spacy_model(user_topic)[0].lemma_
        candidate_words = [self.spacy_model(word)[0].lemma_ for word in candidate_words if 
                        self.spacy_model(word)[0].lemma_ != user_topic_lemma]

        candidate_words = [word for word in candidate_words if len(word) <= max_len]

        return candidate_words

    def get_candidates(self, user_topic, max_len=10):
        user_topic = user_topic.lower()
        candidate_words = [word for word, score in self.glove_model.similar_by_word(user_topic, topn=20)]
        return self.__filter_candidate_words(user_topic, candidate_words, max_len)

if __name__ == "__main__":
    import yaml 

    with open('config.yaml', 'r') as f:
        configs = yaml.safe_load(f)

    answer_generator = AnswerGenerator(configs)
    answers = answer_generator.get_candidates(user_topic="apple", max_len=10)
    print(answers)

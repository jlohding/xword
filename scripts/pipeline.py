import yaml
import gensim.downloader as api
import spacy
import google.generativeai as genai


class LLM:
    def __init__(self, configs):
        genai.configure(api_key=configs["api_key"]["google"])
        self.llm = genai.GenerativeModel(configs["models"]["gemini"])
    
    def generate_clue(self, answer, recursion=0):        
        prompt = f"Generate one clever and short clue for a KEYWORD. Make sure not to include the keyword in the clue. KEYWORD: {answer}"
        response = self.llm.generate_content(prompt)
        
        try:
            clue = response.text
            return clue
        except:
            if recursion >= 3:
                raise Exception("Recursion limit")
            return self.generate_clue(answer,recursion+1)


    def guess_answer(self, clue, answer, recursion=0):
        prompt = f"Using the given CLUE, guess the word. This word has {len(answer)} letters. Only return one word. CLUE: {clue}"
        response = self.llm.generate_content(prompt)
        
        try:
            guess = response.text
            return guess
        except:
            if recursion >= 3:
                raise Exception("Recursion limit")
            return self.guess_answer(clue, answer, recursion+1)

    def improve_clue(self, clue, answer, recursion=0):
        prompt = f"Improve the given CLUE for the ANSWER, without including the keyword in the new clue. ANSWER: {answer}. CLUE: {clue}. "
        response = self.llm.generate_content(prompt)
        
        try:
            new_clue = response.text
            return new_clue
        except:
            if recursion >= 3:
                raise Exception("Recursion limit")
            return self.improve_clue(clue, answer, recursion+1)

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


class Pipeline:
    def __init__(self, configs):
        self.configs = configs
        self.answer_generator = AnswerGenerator(configs)
        self.llm = LLM(configs)

    def main(self, user_topic, max_word_len, n_words):
        candidates = self.answer_generator.get_candidates(user_topic, max_word_len)
        candidates = list(dict.fromkeys(candidates)) # unique

        results = {}
        for candidate in candidates[:n_words]:
            clue = self.llm.generate_clue(candidate)

            # GUESS 
            
            # IMPROVE

            results[candidate] = clue

        return results

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        configs = yaml.safe_load(f)

    pipeline = Pipeline(configs)
    results = pipeline.main(user_topic="apple", max_word_len=5, n_words=25)
    print(results)
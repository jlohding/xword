import yaml
import re
import argparse
from agent import AgentSystem
from answer import AnswerGenerator
from crossword import Crossword


def main(USER_INPUT):
    with open('config.yaml', 'r') as f:
        configs = yaml.safe_load(f) 
    
    answer_generator = AnswerGenerator(configs)
    agent = AgentSystem(configs).set_logging(False)

    answers = answer_generator.get_candidates(user_topic=USER_INPUT, max_len=10)
    word_list = []
    for answer in set(answers):
        clue = agent.generate_clue(answer)
        clean_sentence = re.match(r"(\d*)?([^\(]*)(\(P\d+\))?", clue.clue).group(2).strip()
        word_list.append(
            (answer.upper(), clean_sentence+f" ({len(answer)})")
        )

    print(f"User Input Topic: {USER_INPUT.upper()}")
    cross = Crossword(10, 10, "-", 2000, word_list)
    cross.compute_crossword(time_permitted=1.)
    print(len(cross.current_word_list), 'out of', len(word_list), "words generated used\n")
    print(cross.solution())
    across, down = cross.legend()

    print("ACROSS:")
    for x in across:
        print(x)
    print("DOWN:")
    for x in down:
        print(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("topic")
    args = parser.parse_args()
    main(args.topic)


    


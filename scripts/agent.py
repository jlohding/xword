import yaml
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

class Clue(BaseModel):
    clue: str = Field(description="The clue for the answer")

class Guess(BaseModel):
    guess: str = Field(description="The guess based on the clue")

class Prompts:
    clue_prompt = ChatPromptTemplate.from_template(
    """
    Generate a short clever and cryptic crossword clue for a KEYWORD.
    Make sure not to include the keyword in the clue. KEYWORD: {answer}
    """)

    guess_prompt = ChatPromptTemplate.from_template(
        """
        Using the given CLUE, guess the word. 
        This word has {num_letters} letters. 
        Only return one word. CLUE: {clue}
        """)

    debug_prompt = ChatPromptTemplate.from_template(
        """
        There was an incorrect GUESS for the given CLUE. The correct answer is KEYWORD. 
        Let's think step by step why the previous CLUE was bad, and improve on it logically. 
        Do not include the KEYWORD in the new clue.
        Make sure the new clue is short, clever and cryptic.
        Make sure the new clue makes logical sense.
        Return the new improved crossword clue.
        GUESS: {guess} KEYWORD: {answer}. CLUE: {clue}.
        """)

class AgentSystem:
    def __init__(self, configs):
        self.logging = configs["agent"]["logging"]
        self.timeout = configs["agent"]["timeout"]

        if configs["agent"]["model"] == "google":
            self.model = ChatVertexAI(model=configs["models"]["gemini"], temperature=1)
        elif configs["agent"]["model"] == "openai":
            self.model = ChatOpenAI(
                model=configs["models"]["openai"], temperature=1,
                api_key=configs["api_key"]["openai"])


        self.clue_chain = Prompts.clue_prompt | self.model.with_structured_output(Clue)
        self.guess_chain = Prompts.guess_prompt | self.model.with_structured_output(Guess)
        self.debug_chain = Prompts.debug_prompt | self.model.with_structured_output(Clue)
    
    def __log(self, msg):
        if self.logging:
            print(msg)

    def __check_answer(self, answer: str, guess: Guess):
        return guess.guess.upper() == answer.upper()

    def set_logging(self, to_set: bool):
        self.logging = to_set
        return self

    def generate_clue(self, answer: str):
        self.__log(f"Generating clues for ANSWER: {answer}")
        clue = self.clue_chain.invoke({"answer": answer})
        self.__log(clue)

        timeout = 0
        while timeout < self.timeout:
            timeout += 1

            guess = self.guess_chain.invoke({"num_letters": len(answer), "clue": clue})
            self.__log(guess)

            if self.__check_answer(answer, guess):
                break 
            else:
                clue = self.debug_chain.invoke({"guess": guess, "answer": answer, "clue": clue})
                self.__log(clue)
            
        return clue


if __name__ == "__main__":
    xword = AgentSystem(configs)
    xword.set_logging(True)
    xword.generate_clue("apple")

# Srednja maksimalna naucena vrednost
# Klasifikuj naucene paterne 0-199, 200-349, 350-499, 500, 500+2
# Koliko agenata je dostiglo 500
# Medijan min-max (od max naucenog)
# Ako je dostigao 500 u kojoj epizodi
# Prosecan broj epizoda do sopstvenog maksimuma
import os
import json


class ResearchExample:
    NOT_LEARNED = 0
    BASIC_AGENT = 1
    ADVANCED_AGENT = 2
    PSEUDO_MAX = 3
    GLOBAL_MAX = 4

    def __init__(self, max_value, max_reached_count, max_reached: bool, max_episode_num, agent_class=-1):
        self.max_reached_count = max_reached_count
        self.max_value = max_value
        self.max_reached = max_reached
        self.max_episode_num = max_episode_num
        self.agent_class = -1
        if agent_class == -1:
            self.set_agent_class()
        else:
            self.agent_class = agent_class

    def set_agent_class(self):
        if self.max_reached_count >= 2:
            self.agent_class = ResearchExample.GLOBAL_MAX
            return

        values = [0, 150, 350, 500]
        if self.max_value < values[1]:
            self.agent_class = ResearchExample.NOT_LEARNED
        elif values[1] <= self.max_value < values[2]:
            self.agent_class = ResearchExample.BASIC_AGENT
        elif values[2] <= self.max_value < values[3]:
            self.agent_class = ResearchExample.ADVANCED_AGENT
        elif self.max_value == values[3]:
            self.agent_class = ResearchExample.PSEUDO_MAX


class Research:

    def __init__(self, label):
        self.label = label
        self.raw_data = []
        self.research_examples = []
        self.research = {}

    def process_data(self, example_data: dict):
        reward_list = example_data["rewards"]
        max_value = 0
        max_reached_count = 0
        max_reached = False
        max_episode_num = -1
        for ep_num, reward in enumerate(reward_list):
            if reward > max_value:
                max_value = reward
                max_episode_num = ep_num + 1

            if reward == 500:
                max_reached = True
                max_reached_count += 1

        research_example = ResearchExample(max_value=max_value, max_reached_count=max_reached_count,
                                           max_reached=max_reached, max_episode_num=max_episode_num)

        self.research_examples.append(research_example)
        self.raw_data.append(example_data)

    def process(self):
        # Srednja maksimalna naucena vrednost
        # Klasifikuj naucene paterne 0-199, 200-349, 350-499, 500, 500+2
        # Koliko agenata je dostiglo 500
        # Medijan min-max (od max naucenog)
        # Ako je dostigao 500 u kojoj epizodi
        # Prosecan broj epizoda do sopstvenog maksimuma
        avg_max = 0
        per_class = {i: 0 for i in range(5)}
        count_max = 0

        med_min = 500
        med_max = 0

        med_max_episode_index = 0
        med_min_episode_index = 40

        avg_max_episode_index = 0

        num_examples = 0
        for example in self.research_examples:
            example: ResearchExample
            num_examples += 1
            if example.max_reached:
                count_max += 1

            per_class[example.agent_class] += 1

            if example.max_value < med_min:
                med_min = example.max_value

            if example.max_value > med_max:
                med_max = example.max_value

            if example.max_episode_num < med_min_episode_index:
                med_min_episode_index = example.max_episode_num

            if example.max_episode_num > med_max_episode_index:
                med_max_episode_index = example.max_episode_num

            avg_max += example.max_value
            avg_max_episode_index += example.max_episode_num


        avg_max = avg_max / num_examples
        avg_max_episode_index = avg_max_episode_index / num_examples

        self.research = {
            "label": self.label,
            "avg_max": avg_max,
            "median_max": med_max,
            "median_min": med_min,
            "avg_max_episode_index": avg_max_episode_index,
            "med_max_episode_index": med_max_episode_index,
            "med_min_episode_index": med_min_episode_index,
            "classes": per_class,
            "num_example": num_examples
        }

        research_string = ""
        for val in self.research.values():
            if type(val) != dict:
                research_string += "{}\t".format(val)
            else:
                for ind, v in val.items():
                    research_string += "{}\t".format(v)

                research_string += "{}\t".format(val[3] + val[4])
        print(research_string)
        self.research = {
            "label": self.label,
            "avg_max": avg_max,
            "median_max": med_max,
            "median_min": med_min,
            "avg_max_episode_index": avg_max_episode_index,
            "med_max_episode_index": med_max_episode_index,
            "med_min_episode_index": med_min_episode_index,
            "research_string": research_string,
            "classes": per_class,
            "num_example": num_examples
        }


class ParameterResearch:
    def __init__(self, raw_data_path, save_research_path, parameter="pp"):
        self.parameter = parameter
        self.raw_data_path = raw_data_path
        self.save_research_path = save_research_path
        self.parameter_research = []

    def research(self):

        log_dirs = os.listdir(self.raw_data_path)
        for log_dir in log_dirs:
            label = log_dir.split("_")[0]
            label_path = os.path.join(self.raw_data_path, log_dir)
            data_path = os.path.join(label_path, "data")
            self.research_data(data_path, label, self.save_research_path)

        json.dump(self.parameter_research, open(os.path.join(self.save_research_path, self.parameter + "_research.json"), "w"), indent=4)

    def research_data(self, data_path, label, save_research_path):
        data_json_files = os.listdir(data_path)
        research = Research(label=label)
        for data_json_file in data_json_files:
            example_data = json.load(open(os.path.join(data_path, data_json_file), "r"))
            research.process_data(example_data=example_data)

        research.process()

        save_research_path = os.path.join(save_research_path, label + ".json")
        json.dump(research.research, open(save_research_path, "w"), indent=4)

        self.parameter_research.append(research.research)


gamma_research = ParameterResearch(raw_data_path="data/research/gamma",
                                   save_research_path="data/research/processed_data",
                                   parameter="gamma")

gamma_research.research()

epsilon_d_research = ParameterResearch(raw_data_path="data/research/epsilon-decay",
                                       save_research_path="data/research/processed_data",
                                       parameter="epsilon-decay")

epsilon_d_research.research()


learning_r_research = ParameterResearch(raw_data_path="data/research/learning-rate",
                                        save_research_path="data/research/processed_data",
                                        parameter="learning-rate")

learning_r_research.research()

learning_r_research = ParameterResearch(raw_data_path="data/research/gam2ma",
                                        save_research_path="data/research/processed_data",
                                        parameter="gam2ma")

learning_r_research.research()
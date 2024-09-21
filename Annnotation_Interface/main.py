import gradio as gr
import numpy as np
import json
import sys

def jsonline_reader(path):
    data = []
    with open(path, encoding="utf-8") as reader:
        for row in reader:
            data.append(json.loads(row))
    return data

def jsonline_writer(data, path):
    with open(path, 'w', encoding="utf-8") as writer:
        for x in data:
            writer.write(json.dumps(x, ensure_ascii=False) + '\n')
    writer.close()

def display_close_message():
    return gr.Close.get_script(code="alert('Please close this window manually.');")

class Evaluator():
    def __init__(self, stats_path='./stats.jsonl'):
        n = 5
        self.stats_path = stats_path
        self.contents = jsonline_reader(self.stats_path)
        self.idx_list = self.get_idx_list(self.contents)
        if len(self.idx_list) == 0:
            print('All finished!')
            sys.exit(0)

        self.choices = [list(np.arange(3)) for _ in range(n)]

        self.overall_question = "Give an overall score of this response 给这个回复做一个综合评价"
        self.overall_choices = list(np.arange(11))

    def get_idx_list(self, contents):
        idx_list = []
        for i, x in enumerate(contents):
            if x['flags'] == False:
                idx_list.append(i)
        print('Total length', len(idx_list))
        return idx_list

    def process_choices(self, *args):
        results = args[:-1]
        index = args[-1]

        idx = self.idx_list[index]
        self.contents[idx]['flags'] = True
        self.contents[idx]['evaluation_results'] = results

        print(f'{idx}({index}) response evaluation finished!')
        already_num = self.calculate_already_finished()
        print(f'Progress: {already_num}/{len(self.idx_list)}')

        jsonline_writer(self.contents, self.stats_path)
    
    def calculate_already_finished(self):
        num = 0
        for i in self.idx_list:
            # print(self.contents[i]['flags'])
            if self.contents[i]['flags'] == True:
                num += 1
        return num

    def create_quiz_interface(self):
        init_idx = self.idx_list[0]
        # print(init_idx)
        radios = [gr.Radio(self.choices[i], label=' '.join(self.contents[init_idx]['checklist'][i]), interactive=True) for i in range(len(self.contents[init_idx]['checklist']))]
        overall_radio = gr.Radio(self.overall_choices, label=self.overall_question, interactive=True)
        
        # inputs = radios + [overall_radio]
        return radios, overall_radio

    def generate_html_content(self, idx):
        return '''
        <div style='display: flex; flex-direction: column;'>
            <div style='display: flex;'>
                <div style='width: 50%; padding: 10px; border: 1px solid black; height: 400px; overflow-y: auto;'>
                    <h2>English input</h2>
                    <p>{}</p>
                </div>
                <div style='width: 50%; padding: 10px; border: 1px solid black; height: 400px; overflow-y: auto;'>
                    <h2>English response</h2>
                    <p>{}</p>
                </div>
            </div>
            <div style='display: flex;'>
                <div style='width: 50%; padding: 10px; border: 1px solid black; height: 400px; overflow-y: auto;'>
                    <h2>Chinese input</h2>
                    <p>{}</p>
                </div>
                <div style='width: 50%; padding: 10px; border: 1px solid black; height: 400px; overflow-y: auto;'>
                    <h2>Chinese response</h2>
                    <p>{}</p>
                </div>
            </div>
        </div>
        '''.format(
            self.contents[idx]['instruction_en'].replace('\n', '<br>'), 
            self.contents[idx]['response_en'].replace('\n', '<br>'), 
            self.contents[idx]['instruction_zh'].replace('\n', '<br>'),
            self.contents[idx]['response_zh'].replace('\n', '<br>')
        )

    def update_html(self, next_index):
        idx = self.idx_list[next_index]
        print(f'Change to {idx}({next_index}) response')

        return self.generate_html_content(idx)

    def update_radios(self, next_index):
        idx = self.idx_list[next_index]
        # Assuming dynamic labels based on the index
        new_questions = ['\n'.join(self.contents[idx]['checklist'][i]) for i in range(len(self.contents[idx]['checklist']))]

        return [gr.update(label=new_questions[i]) for i in range(len(new_questions))]

    def check_exist(self, idx):
        if idx >= len(self.idx_list):
            print('You have finished, reopen the program to check if non response left.')
            sys.exit(0)

    def start(self):
        with gr.Blocks() as main:
            self.current_index = gr.State(0)
            with gr.Row():
                with gr.Column(scale=2):
                    init_idx = self.idx_list[0]
                    html_display = gr.HTML(self.generate_html_content(init_idx))
                with gr.Column(scale=1):
                    radios, overall_radio = self.create_quiz_interface()
                    inputs = radios + [overall_radio]
                    submit_button = gr.Button("Submit")
                    submit_button.click(fn=self.process_choices, inputs=inputs + [self.current_index], outputs=None)

                    next_button = gr.Button("Next")
                    next_button.click(fn=lambda idx: idx + 1, inputs=self.current_index, outputs=self.current_index)
                    next_button.click(fn=self.update_html, inputs=self.current_index, outputs=html_display)
                    next_button.click(fn=self.update_radios, inputs=self.current_index, outputs=radios)
                    next_button.click(fn=self.check_exist, inputs=self.current_index, outputs=None)
  
        main.launch()

if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.start()

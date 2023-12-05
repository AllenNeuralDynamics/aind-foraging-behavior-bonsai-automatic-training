import re

from graphviz import Digraph
from dynamic_foraging_curriculum.plot.manager import stage_color_mapper


def _format_lambda_full(string: str):
    return '\n'.join([line.lstrip()   # Remove the first line
                      for line in
                      string.split('\n')[1:]])  # Strip spaces


def _format_lambda_description(string: str):
    # Define the pattern to search for 'and', 'or', or ','
    pattern = r'(\band\b|\bor\b|,)'
    # Replace the found pattern with itself followed by a newline
    wrapped_text = re.sub(pattern, r'\1\n', string)
    return wrapped_text


def draw_curriculum_diagram(curriculum: 'Curriculum',
                            ):
    """Generate dot file by graphviz 

    Args:
        curriculum (Curriculum): _description_
    """

    # Script data extracted from the user's script
    stages = curriculum.parameters.keys()

    # Create Digraph object
    dot = Digraph(comment='Curriculum for Dynamic Foraging - Coupled Baiting')

    # From bottom to top layout
    dot.attr(rankdir='TB')

    # Add nodes (stages)
    for stage in stages:
        dot.node(name=stage.name,
                 label=stage.name,
                 shape='ellipse',
                 style='filled',
                 fillcolor=stage_color_mapper[stage.name],
                 tooltip=curriculum.parameters[stage].description)

    # Add the Graduated node
    dot.node(name='GRADUATED',
             label='GRADUATED',
             shape='box',
             style='filled',
             fillcolor=stage_color_mapper['GRADUATED'])

    # Add transitions
    for stage, stage_transitions in curriculum.curriculum.items():
        for transition_rule in stage_transitions.transition_rules:
            lambda_full_string = _format_lambda_full(
                transition_rule.condition)
            lambda_descr_string = _format_lambda_description(
                transition_rule.condition_description)
            style = 'solid' if transition_rule.decision.name == 'PROGRESS' else 'dashed'
            color = 'black' if transition_rule.decision.name == 'PROGRESS' else 'grey'
            
            dot.edge(tail_name=stage.name,
                     head_name=transition_rule.to_stage.name,
                     minlen='2',
                     **(dict(headlabel=lambda_descr_string, label='      ') 
                        if transition_rule.decision.name == 'PROGRESS' 
                        else dict(taillabel=lambda_descr_string)),
                     edgetooltip=lambda_full_string,
                     tailtooltip=lambda_full_string,
                     headtooltip=lambda_full_string,
                     style=style,
                     color=color,
                     fontcolor=color,
                     )

    # # Visualize the graph
    # dot.render('dynamic_foraging_curriculum', format='png', cleanup=True)

    return dot


if __name__ == '__main__':
    import json

    from dynamic_foraging_curriculum.schema.curriculum import DynamicForagingCurriculum, TrainingStage

    with open("/root/capsule/code/dynamic_foraging_curriculum/curriculums/curriculum_Coupled Baiting_0.1_1.0.json", "r") as f:
        loaded_json = json.load(f)

    loaded_curriculum = DynamicForagingCurriculum(**loaded_json)
    dot = draw_curriculum_diagram(loaded_curriculum)

    dot.render('dynamic_foraging_curriculum.svg', format='svg')

    print(dot)

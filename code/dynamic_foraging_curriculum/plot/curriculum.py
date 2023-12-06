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
                            ) -> 'dot file':
    """Generate stage transition rules by graphviz 

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
                     **(dict(headlabel=lambda_descr_string,
                             headtooltip=lambda_full_string,
                             taillabel='',
                             tailtooltip=' ',
                             label='      ')
                        if transition_rule.decision.name == 'PROGRESS'
                        else dict(headlabel='',
                                  headtooltip=' ',
                                  taillabel=lambda_descr_string,
                                  tailtooltip=lambda_full_string,
                                  )),
                     edgetooltip=transition_rule.decision.name,
                     style=style,
                     color=color,
                     fontcolor=color,
                     penwidth='2'
                     )

    # # Visualize the graph
    # dot.render('dynamic_foraging_curriculum', format='png', cleanup=True)

    return dot


def draw_parameter_table(curriculum: 'Curriculum',
                         ) -> 'dot file':
    """Generate detailed parameter table by graphviz
    with change of parameters highlighted in green
    """

    dot = Digraph('G')

    # Graph attributes to control the overall appearance
    dot.attr(rankdir='TB', splines='ortho')
    dot.attr(nodesep='0.1', ranksep='0.0')
    dot.attr('node',
             shape='box',
             style='filled',
             fillcolor='lightgrey',
             width='0', height='0.2')

    # Add nodes for parameters
    paras = []
    for n_stage, (stage, task_schema) in enumerate(curriculum.parameters.items()):
        dict_paras = task_schema.to_GUI_format()
        paras.append(dict_paras)  # Cache the parameters
        dict_paras = {k: v for k, v in dict_paras.items() 
                      if k not in ('task', 'task_schema_version', 'curriculum_version',
                                      'training_stage', 'description', 'UncoupledReward')}

        for n_para, (para_name, para_value) in enumerate(dict_paras.items()):
            # Hightlight the changed parameters
            if n_stage > 0 and para_value != paras[-2][para_name]:
                fillcolor = 'lightgreen'
            else:
                fillcolor = 'lightgrey'

            # Add cell of the table as a node
            # For parameters, (row, column) starting from (1, 1)
            dot.node(name=f'cell_{n_para+1}_{n_stage+1}',
                     fillcolor=fillcolor,
                     label=str(para_value))

    # Add first column as paras names
    for n_para, (para_name, _) in enumerate(dict_paras.items()):
        dot.node(name=f'cell_{n_para+1}_0',
                 fillcolor='lightgrey',
                 label=para_name,
                 # Retrieve description from the schema
                 tooltip=task_schema.schema()['properties'][para_name]['title']
                 )
        
    # Add first row as stage names
    for n_stage, (stage, _) in enumerate(curriculum.parameters.items()):
        dot.node(name=f'cell_0_{n_stage+1}',
                 fillcolor=stage_color_mapper[stage.name],
                 label=stage.name,
                 tooltip=curriculum.parameters[stage].description)


    # Invisible edges to position nodes in a grid
    for row in range(0, n_para+2):
        for col in range(0, n_stage+2):
            if row + 1 <= n_para:
                dot.edge(f'cell_{row}_{col}', f'cell_{row+1}_{col}')
            if col + 1 <= n_stage:
                dot.edge(f'cell_{row}_{col}', f'cell_{row}_{col+1}')


    # Using subgraphs to align nodes in rows
    for row in range(0, n_para+2):
        with dot.subgraph(name=f'row_{row}') as row_graph:
            row_graph.attr(rank='same')
            for col in range(0, n_stage+1):
                row_graph.node(f'cell_{row}_{col}')

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

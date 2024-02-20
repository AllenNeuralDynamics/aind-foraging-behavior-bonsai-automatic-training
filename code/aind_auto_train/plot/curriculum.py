import re

from graphviz import Digraph
from matplotlib import pyplot as plt
import matplotlib

def get_stage_color_mapper(stage_list):
    # Mapping stages to colors from red to green, return rgb values
    # Interpolate between red and green using the number of stages
    cmap = plt.cm.get_cmap('RdYlGn', 100)
    stage_color_mapper = {
        stage: matplotlib.colors.rgb2hex(
            cmap(i / (len(stage_list) - 1))) 
        for i, stage in enumerate(stage_list)
    }
    return stage_color_mapper

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


def draw_diagram_rules(curriculum):
    """Generate stage transition rules by graphviz 

    Args:
        curriculum (Curriculum): _description_
    """

    stages = curriculum.parameters.keys()
    stage_color_mapper = get_stage_color_mapper(
        [s.name for s in list(stages)] + ['GRADUATED']
    )

    # Create Digraph object
    dot = Digraph(comment='Curriculum for Dynamic Foraging - Coupled Baiting')
    dot.attr(label=f"{curriculum.curriculum_name} "
                   f"(v{curriculum.curriculum_version} "
                   f"@ schema v{curriculum.curriculum_schema_version})\n"
                   f'{curriculum.curriculum_description}',
             labelloc='t',
             fontsize='17'
             )

    # From bottom to top layout
    dot.attr(rankdir='TB')

    # Add nodes (stages)
    for stage in stages:
        dot.node(name=stage.name,
                 label=f'{stage.name}\n{curriculum.parameters[stage].task}',
                 shape='ellipse',
                 style='filled',
                 fillcolor=stage_color_mapper[stage.name],
                 tooltip=curriculum.parameters[stage].description + \
                     f'\nstage_task = {curriculum.parameters[stage].task}')

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
    # dot.render('aind_auto_train', format='png', cleanup=True)

    return dot


def draw_diagram_paras(curriculum,
                       min_value_width=1,
                       min_var_name_width=2,
                       fontsize=12,
                       ):
    """Generate detailed parameter table by graphviz
    with change of parameters highlighted in green
    """

    stages = curriculum.parameters.keys()
    stage_color_mapper = get_stage_color_mapper(
        [s.name for s in list(stages)] + ['GRADUATED']
    )

    dot = Digraph('G')

    # Graph attributes to control the overall appearance
    dot.attr(rankdir='TB', splines='ortho')
    dot.attr(nodesep='0.1', ranksep='0.0')
    dot.attr('node',
             shape='box',
             style='filled',
             fillcolor='lightgrey',
             fontsize=str(fontsize),
             width=str(min_var_name_width),
             height='0.2'
             )

    # Add nodes for parameters
    paras = []
    for i_stage, (stage, task_schema) in enumerate(curriculum.parameters.items()):
        dict_paras = task_schema.to_GUI_format()
        paras.append(dict_paras)  # Cache the parameters
        dict_paras = {k: v for k, v in dict_paras.items()
                      if k not in ('task', 'task_url', 'task_schema_version', 'curriculum_version',
                                   'training_stage', 'description')}

        for i_para, (para_name, para_value) in enumerate(dict_paras.items()):
            # Hightlight the changed parameters
            if i_stage > 0 and para_value != paras[-2][para_name]:
                fillcolor = 'darkseagreen'
            else:
                fillcolor = 'lightgrey'

            # Add cell of the table as a node
            # For parameters, (row, column) starting from (1, 1)
            dot.node(name=f'cell_{i_para+1}_{i_stage+1}',
                     fillcolor=fillcolor,
                     label=str(para_value),
                     width=str(min_value_width),
                     tooltip=f'{para_name} @ {stage.name}')

    # Add first column as paras names
    for i_para, (para_name, _) in enumerate(dict_paras.items()):
        dot.node(name=f'cell_{i_para+1}_0',
                 fillcolor='lightgrey',
                 label=para_name,
                 # Retrieve description from the schema
                 tooltip=task_schema.schema()['properties'][para_name]['title']
                 )

    # Add first row as stage names
    for i_stage, (stage, _) in enumerate(curriculum.parameters.items()):
        dot.node(name=f'cell_0_{i_stage+1}',
                 fillcolor=stage_color_mapper[stage.name],
                 label=stage.name,
                 tooltip=curriculum.parameters[stage].description + \
                     f'\nstage_task = {curriculum.parameters[stage].task}',
                 width=str(min_value_width),
        )

    # Invisible edges to position nodes in a grid
    n_cols = len(curriculum.parameters) + 1
    n_rows = len(dict_paras) + 1

    for col in range(0, n_cols):
        for row in range(0, n_rows):
            if col == 0 and row == 0: 
                continue
            if row < n_rows - 1:
                dot.edge(f'cell_{row}_{col}', f'cell_{row+1}_{col}', style='invis')
            if col < n_cols - 1:
                dot.edge(f'cell_{row}_{col}', f'cell_{row}_{col+1}', style='invis')

    # Using subgraphs to align nodes in rows
    for row in range(0, n_rows):
        with dot.subgraph(name=f'row_{row}') as row_graph:
            row_graph.attr(rank='same')
            for col in range(0, n_cols):
                if row == 0 and col == 0:
                    continue
                row_graph.node(f'cell_{row}_{col}')

    return dot


if __name__ == '__main__':
    import json

    from aind_auto_train.schema.curriculum import DynamicForagingCurriculum

    with open("/root/capsule/code/aind_auto_train/curriculums/curriculum_Coupled Baiting_0.1_1.0.json", "r") as f:
        loaded_json = json.load(f)

    loaded_curriculum = DynamicForagingCurriculum(**loaded_json)
    dot = draw_diagram_rules(loaded_curriculum)

    dot.render('aind_auto_train.svg', format='svg')

    print(dot)

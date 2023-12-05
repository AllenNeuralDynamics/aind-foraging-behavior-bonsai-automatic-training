from graphviz import Digraph


def _format_lambda_description(lambdastr_descr: str):
    return '\n'.join([line.lstrip()   # Remove the first line
                      for line in
                      lambdastr_descr.split('\n')[1:]]) # Strip spaces


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
    dot.attr(randir='BT')

    # Add nodes (stages)
    for stage in stages:
        dot.node(name=stage.name,
                 label=stage.name,
                 tooltip=curriculum.parameters[stage].description)

    # Add edges (transitions)
    for stage, stage_transitions in curriculum.curriculum.items():
        for transition_rule in stage_transitions.transition_rules:
            lambda_descr = _format_lambda_description(
                transition_rule.condition)

            dot.edge(tail_name=stage.name,
                     head_name=transition_rule.to_stage.name,
                     label=transition_rule.condition_description,
                     edgetooltip=lambda_descr,
                     labeltooltip=lambda_descr)

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

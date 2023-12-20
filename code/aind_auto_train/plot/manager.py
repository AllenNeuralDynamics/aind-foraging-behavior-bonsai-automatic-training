import numpy as np
import plotly.graph_objects as go

from aind_auto_train.schema.curriculum import TrainingStage

# Define color scale - mapping stages to colors from red to green
# TODO: make this flexible
stage_color_mapper = {
    TrainingStage.STAGE_1.name: 'red',
    TrainingStage.STAGE_2.name: 'orange',
    TrainingStage.STAGE_3.name: 'yellow',
    TrainingStage.STAGE_FINAL.name: 'lightgreen',
    TrainingStage.GRADUATED.name: 'green'
}


def plot_manager_all_progress(manager: 'AutoTrainManager',
                              if_show_fig=True
                              ):
    # %%
    # Plot the training history of a mouse
    df_manager = manager.df_manager

    # Preparing the scatter plot
    traces = []
    for n, subject_id in enumerate(df_manager['subject_id'].unique()):
        df_subject = df_manager[df_manager['subject_id'] == subject_id]
        # Get h2o if available
        if 'h2o' in manager.df_behavior:
            h2o = manager.df_behavior[
                manager.df_behavior['subject_id'] == subject_id]['h2o'].iloc[0]
        else:
            h2o = None

        # Set fill color (none if not closed loop)
        open_loop_ids = df_subject.if_closed_loop == False
        color_actual = df_subject['current_stage_actual'].map(
            stage_color_mapper)
        color_actual[open_loop_ids] = 'lightgrey'
        stage_actual = df_subject.current_stage_actual.values
        stage_actual[open_loop_ids] = 'unknown (open loop)'

        traces.append(go.Scattergl(
            x=df_subject['session'],
            y=[n] * len(df_subject),
            mode='markers',
            marker=dict(
                size=10,
                line=dict(
                    width=2,
                    color=df_subject['current_stage_suggested'].map(
                        stage_color_mapper)
                ),
                color=color_actual,
                # colorbar=dict(title='Training Stage'),
            ),
            name=f'Mouse {subject_id}',
            hovertemplate=(f"<b>Subject {subject_id} ({h2o})"
                           "<br>Session %{x}, %{customdata[4]}</b>"
                           "<br>Curriculum: <b>%{customdata[7]}_v%{customdata[8]}</b>"
                           "<br>Suggested: <b>%{customdata[0]}</b>"
                           "<br>Actual: <b>%{customdata[1]}</b>"
                           "<br>Session task: <b>%{customdata[6]}</b>"
                           "<br>foraging_eff = %{customdata[2]}"
                           "<br>finished_trials = %{customdata[3]}"
                           "<br>Decision = <b>%{customdata[5]}</b>"
                           "<extra></extra>"),
            customdata=np.stack(
                (df_subject.current_stage_suggested,
                 stage_actual,
                 np.round(df_subject.foraging_efficiency, 3),
                 df_subject.finished_trials,
                 df_subject.session_date,
                 df_subject.decision,
                 df_subject.task,
                 df_subject.curriculum_task,
                 df_subject.curriculum_version,
                 ), axis=-1),
            showlegend=False
        )
        )

        # Add "x" for open loop sessions
        traces.append(go.Scattergl(
            x=df_subject['session'][open_loop_ids],
            y=[n] * len(df_subject['session'][open_loop_ids]),
            mode='markers',
            marker=dict(
                size=5,
                symbol='x-thin',
                color='black',
                line_width=1,
            ),
            showlegend=False,
        )
        )

    # Create the figure
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f'Training Progress of All Mice ({manager.manager_name}, curriculum_task = {manager.df_manager.curriculum_task[0]})',
        xaxis_title='Session',
        yaxis_title='Mouse',
        height=1200,
    )

    # Set subject_id as y axis label
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(0, n + 1),  # Original y-axis values
            ticktext=df_manager['subject_id'].unique()  # New labels
        )
    )

    # Show the plot
    if if_show_fig:
        fig.show()

    # %%
    return fig

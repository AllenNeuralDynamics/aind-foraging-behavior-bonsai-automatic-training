import numpy as np
import plotly.graph_objects as go

from aind_auto_training.schema.curriculum import TrainingStage

# Define color scale - mapping stages to colors from red to green
# TODO: make this flexible
stage_color_mapper = {
    TrainingStage.STAGE_1.name: 'red',
    TrainingStage.STAGE_2.name: 'orange',
    TrainingStage.STAGE_3.name: 'yellow',
    TrainingStage.STAGE_FINAL.name: 'lightgreen',
    TrainingStage.GRADUATED.name: 'green'
}

def plot_manager_all_progress(manager: 'CurriculumManager',
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

        trace = go.Scattergl(
            x=df_subject['session'],
            y=[n] * len(df_subject),
            mode='markers',
            marker=dict(
                size=10,
                line=dict(width=1, color='black'),
                color=df_subject['current_stage_suggested'].map(
                    stage_color_mapper),
                # colorbar=dict(title='Training Stage'),
            ),
            name=f'Mouse {subject_id}',
            hovertemplate=(f"<b>Subject {subject_id} ({h2o})"
                           "<br>Session %{x}"
                           "<br>%{customdata[0]}</b>"
                           "<br>foraging_eff = %{customdata[1]}"
                           "<br>finished_trials = %{customdata[2]}"
                           "<extra></extra>"),
            customdata=np.stack(
                (df_subject.current_stage_suggested,
                    df_subject.foraging_efficiency,
                    df_subject.finished_trials), axis=-1),
            showlegend=False
        )
        traces.append(trace)

    # Create the figure
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f'Training Progress of All Mice ({manager.manager_name}, task = {manager.df_manager.task[0]})',
        xaxis_title='Session',
        yaxis_title='Mouse',
        height=1200
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

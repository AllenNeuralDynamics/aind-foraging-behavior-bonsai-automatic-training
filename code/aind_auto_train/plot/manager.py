from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import pandas as pd

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
                              x_axis: ['session', 'date',
                                       'relative_date'] = 'session',
                              sort_by: ['subject_id', 'first_date',
                                        'last_date', 'progress_to_graduated'] = 'subject_id',
                              sort_order: ['ascending',
                                           'descending'] = 'descending',
                              if_show_fig=True
                              ):
    # %%
    # Set default order
    df_manager = manager.df_manager.sort_values(by=['subject_id', 'session'],
                                                ascending=[sort_order == 'ascending', False])

    # Sort mice
    if sort_by == 'subject_id':
        subject_ids = df_manager.subject_id.unique()
    elif sort_by == 'first_date':
        subject_ids = df_manager.groupby('subject_id').session_date.min().sort_values(
            ascending=sort_order == 'ascending').index
    elif sort_by == 'last_date':
        subject_ids = df_manager.groupby('subject_id').session_date.max().sort_values(
            ascending=sort_order == 'ascending').index
    elif sort_by == 'progress_to_graduated':
        manager.compute_stats()
        df_stats = manager.df_manager_stats
        
        # Sort by 'first_entry' of GRADUATED
        subject_ids = df_stats.reset_index().set_index(
            'subject_id'
        ).query(
            f'current_stage_actual == "GRADUATED"'
        )['first_entry'].sort_values(
            ascending=sort_order != 'ascending').index.to_list()
        
        # Append subjects that have not graduated
        subject_ids = subject_ids + [s for s in df_manager.subject_id.unique() if s not in subject_ids]
        
    else:
        raise ValueError(
            f'sort_by must be in {["subject_id", "first_date", "last_date", "progress"]}')

    # Preparing the scatter plot
    traces = []
    for n, subject_id in enumerate(subject_ids):
        df_subject = df_manager[df_manager['subject_id'] == subject_id]
        # Get h2o if available
        if 'h2o' in manager.df_behavior:
            h2o = manager.df_behavior[
                manager.df_behavior['subject_id'] == subject_id]['h2o'].iloc[0]
        else:
            h2o = None

        # Handle open loop sessions
        open_loop_ids = df_subject.if_closed_loop == False
        color_actual = df_subject['current_stage_actual'].map(
            stage_color_mapper)
        color_actual[open_loop_ids] = 'lightgrey'
        stage_actual = df_subject.current_stage_actual.values
        stage_actual[open_loop_ids] = 'unknown (open loop)'

        # Select x
        if x_axis == 'session':
            x = df_subject['session']
        elif x_axis == 'date':
            x = df_subject['session_date']
        elif x_axis == 'relative_date':
            x = pd.to_datetime(df_subject['session_date'])
            x = (x - x.min()).dt.days
        else:
            raise ValueError(
                f"x_axis can only be in ['session', 'date', 'relative_date']")

        traces.append(go.Scattergl(
            x=x,
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
            hovertemplate=(f"<b>Subject {subject_id} ({h2o})</b>"
                           "<br><b>Session %{customdata[0]}, %{customdata[1]}</b>"
                           "<br>Curriculum: <b>%{customdata[2]}_v%{customdata[3]}</b>"
                           "<br>Suggested: <b>%{customdata[4]}</b>"
                           "<br>Actual: <b>%{customdata[5]}</b>"
                           "<br>Session task: <b>%{customdata[6]}</b>"
                           "<br>foraging_eff = %{customdata[7]}"
                           "<br>finished_trials = %{customdata[8]}"
                           "<br>Decision = <b>%{customdata[9]}</b>"
                           "<br>Next suggested: <b>%{customdata[10]}</b>"
                           "<extra></extra>"),
            customdata=np.stack(
                (df_subject.session,
                 df_subject.session_date,
                 df_subject.curriculum_task,
                 df_subject.curriculum_version,
                 df_subject.current_stage_suggested,
                 stage_actual,
                 df_subject.task,
                 np.round(df_subject.foraging_efficiency, 3),
                 df_subject.finished_trials,
                 df_subject.decision,
                 df_subject.next_stage_suggested,
                 ), axis=-1),
            showlegend=False
        )
        )

        # Add "x" for open loop sessions
        traces.append(go.Scattergl(
            x=x[open_loop_ids],
            y=[n] * len(x[open_loop_ids]),
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
        title=f"Training progress ({manager.manager_name}, "
              f"curriculum_task = {manager.df_manager.curriculum_task[0]})",
        xaxis_title=x_axis,
        yaxis_title='Mouse',
        height=1200,
    )

    # Set subject_id as y axis label
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(0, n + 1),  # Original y-axis values
            ticktext=subject_ids,  # New labels
            autorange='reversed',
        )
    )

    # Show the plot
    if if_show_fig:
        fig.show()

    # %%
    return fig

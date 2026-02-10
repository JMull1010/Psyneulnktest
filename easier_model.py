import timeit
import numpy as np
from psyneulink import *
import KeysAndDoorsWrapper as kad

# Runtime Switches:
RENDER = True
PNL_COMPILE = False
RUN = True

# *********************************************************************************************************************
# *********************************************** CONSTANTS ***********************************************************
# *********************************************************************************************************************

obs_len = 8
num_state_nodes = 8
num_doors = 1
num_keys = 1

# *********************************************************************************************************************
# **************************************  EM COMPOSITION  *************************************************************
# *********************************************************************************************************************

# State input for EM - now expects flat array
state_input = ProcessingMechanism(name='STATE INPUT',
                                  default_variable=[0] * 8)

# Output mechanism for EM
em_output = ProcessingMechanism(name="EM OUTPUT",
                                default_variable=[0, 0, 0, 0, 0, 0, 0])

# Translation constants
empty = -1
none = 0
false = 0
true = 1
red = 1
green = 2
blue = 3
closed = 4
open = 5
key = 1
no_key = 0
h = 1
j = 2
t = 0
certain = 1
not_certain = 0

# EM initialization entries
em_init_entries = [
    # X, Y, Door states, Key states, Holding Key, Key Color, Heaven, Certainty
    # RIGHT, LEFT, UP, DOWN, OPEN, PICKUP, READ
    ([0], [3], [-1], [-1], [false], [none], [none], [certain],
     [1], [0], [0], [0], [0], [0], [0]),
    ([1], [3], [-1], [-1], [false], [none], [none], [certain],
     [1], [0], [0], [0], [0], [0], [0]),
    ([2], [3], [-1], [-1], [false], [none], [none], [certain],
     [1], [0], [0], [0], [0], [0], [0]),
    ([3], [3], [-1], [-1], [false], [none], [none], [certain],
     [0], [0], [1], [0], [0], [0], [0]),
    ([3], [2], [-1], [-1], [false], [none], [none], [certain],
     [0], [0], [1], [0], [0], [0], [0]),
    ([3], [1], [-1], [-1], [false], [none], [none], [certain],
     [0], [1], [0], [0], [0], [0], [0]),
    ([2], [1], [-1], [-1], [false], [none], [none], [certain],
     [0], [1], [0], [0], [0], [0], [0]),
    ([1], [1], [-1], [-1], [false], [none], [none], [certain],
     [0], [1], [0], [0], [0], [0], [0]),
    ([0], [1], [-1], [-1], [false], [none], [none], [certain],
     [0], [0], [1], [0], [0], [0], [0]),
]

# Create EM Composition
instruct_em = EMComposition(
    name="instruct_em",
    memory_template=em_init_entries,
    memory_capacity=10,
    memory_decay_rate=0,
    memory_fill=0.001,
    storage_prob=0.0,
    fields={
        "AGENT X": {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        "AGENT Y": {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        "DOOR STATES": {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        "KEY STATES": {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        "HOLDING KEY": {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        "KEY COLOR": {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        "HEAVEN": {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        "CERTAINTY": {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        "RIGHT": {FIELD_WEIGHT: None, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: True},
        "LEFT": {FIELD_WEIGHT: None, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: True},
        "UP": {FIELD_WEIGHT: None, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: True},
        "DOWN": {FIELD_WEIGHT: None, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: True},
        "OPEN ACTION": {FIELD_WEIGHT: None, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: True},
        "PICKUP": {FIELD_WEIGHT: None, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: True},
        "READ": {FIELD_WEIGHT: None, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: True},
    },
    softmax_choice=ARG_MAX,
    normalize_memories=True,
    enable_learning=False,
    softmax_gain=10.0
)

x_extract = np.zeros((8, 1))
x_extract[0, 0] = 1

y_extract = np.zeros((8, 1))
y_extract[1, 0] = 1

door_extract = np.zeros((8, 1))
door_extract[2, 0] = 1

key_extract = np.zeros((8, 1))
key_extract[3, 0] = 1

holding_extract = np.zeros((8, 1))
holding_extract[4, 0] = 1

color_extract = np.zeros((8, 1))
color_extract[5, 0] = 1

heaven_extract = np.zeros((8, 1))
heaven_extract[6, 0] = 1

certainty_extract = np.zeros((8, 1))
certainty_extract[7, 0] = 1

# Matrices for mapping EM outputs to action vector
right_matrix = np.array([[1, 0, 0, 0, 0, 0, 0]])
left_matrix = np.array([[0, 1, 0, 0, 0, 0, 0]])
up_matrix = np.array([[0, 0, 1, 0, 0, 0, 0]])
down_matrix = np.array([[0, 0, 0, 1, 0, 0, 0]])
open_matrix = np.array([[0, 0, 0, 0, 1, 0, 0]])
pickup_matrix = np.array([[0, 0, 0, 0, 0, 1, 0]])
read_matrix = np.array([[0, 0, 0, 0, 0, 0, 1]])

# Query pathways (state inputs to EM) - using extraction matrices
state_to_em_agent_x = [state_input,
                       MappingProjection(matrix=x_extract,
                                         sender=state_input,
                                         receiver=instruct_em.nodes["AGENT X [QUERY]"],
                                         learnable=False),
                       instruct_em
                       ]
state_to_em_agent_y = [state_input,
                       MappingProjection(matrix=y_extract,
                                         sender=state_input,
                                         receiver=instruct_em.nodes["AGENT Y [QUERY]"],
                                         learnable=False),
                       instruct_em
                       ]
state_to_em_door_states = [state_input,
                           MappingProjection(matrix=door_extract,
                                             sender=state_input,
                                             receiver=instruct_em.nodes["DOOR STATES [QUERY]"],
                                             learnable=False),
                           instruct_em
                           ]
state_to_em_key_states = [state_input,
                          MappingProjection(matrix=key_extract,
                                            sender=state_input,
                                            receiver=instruct_em.nodes["KEY STATES [QUERY]"],
                                            learnable=False),
                          instruct_em
                          ]
state_to_em_holding_key = [state_input,
                           MappingProjection(matrix=holding_extract,
                                             sender=state_input,
                                             receiver=instruct_em.nodes["HOLDING KEY [QUERY]"],
                                             learnable=False),
                           instruct_em
                           ]
state_to_em_key_color = [state_input,
                         MappingProjection(matrix=color_extract,
                                           sender=state_input,
                                           receiver=instruct_em.nodes["KEY COLOR [QUERY]"],
                                           learnable=False),
                         instruct_em
                         ]
state_to_em_heaven = [state_input,
                      MappingProjection(matrix=heaven_extract,
                                        sender=state_input,
                                        receiver=instruct_em.nodes["HEAVEN [QUERY]"],
                                        learnable=False),
                      instruct_em
                      ]
state_to_em_certainty = [state_input,
                         MappingProjection(matrix=certainty_extract,
                                           sender=state_input,
                                           receiver=instruct_em.nodes["CERTAINTY [QUERY]"],
                                           learnable=False),
                         instruct_em
                         ]

zero_to_value = np.zeros((8, 1))

state_to_em_right_value = [state_input,
                           MappingProjection(matrix=zero_to_value,
                                             sender=state_input,
                                             receiver=instruct_em.nodes["RIGHT [VALUE]"],
                                             learnable=False),
                           instruct_em]
state_to_em_left_value = [state_input,
                          MappingProjection(matrix=zero_to_value,
                                            sender=state_input,
                                            receiver=instruct_em.nodes["LEFT [VALUE]"],
                                            learnable=False),
                          instruct_em]
state_to_em_up_value = [state_input,
                        MappingProjection(matrix=zero_to_value,
                                          sender=state_input,
                                          receiver=instruct_em.nodes["UP [VALUE]"],
                                          learnable=False),
                        instruct_em]
state_to_em_down_value = [state_input,
                          MappingProjection(matrix=zero_to_value,
                                            sender=state_input,
                                            receiver=instruct_em.nodes["DOWN [VALUE]"],
                                            learnable=False),
                          instruct_em]
state_to_em_open_value = [state_input,
                          MappingProjection(matrix=zero_to_value,
                                            sender=state_input,
                                            receiver=instruct_em.nodes["OPEN ACTION [VALUE]"],
                                            learnable=False),
                          instruct_em]
state_to_em_pickup_value = [state_input,
                            MappingProjection(matrix=zero_to_value,
                                              sender=state_input,
                                              receiver=instruct_em.nodes["PICKUP [VALUE]"],
                                              learnable=False),
                            instruct_em]
state_to_em_read_value = [state_input,
                          MappingProjection(matrix=zero_to_value,
                                            sender=state_input,
                                            receiver=instruct_em.nodes["READ [VALUE]"],
                                            learnable=False),
                          instruct_em]

state_to_em_right = [instruct_em,
                     MappingProjection(matrix=right_matrix,
                                       sender=instruct_em.nodes["RIGHT [RETRIEVED]"],
                                       receiver=em_output,
                                       learnable=False),
                     em_output
                     ]
state_to_em_left = [instruct_em,
                    MappingProjection(matrix=left_matrix,
                                      sender=instruct_em.nodes["LEFT [RETRIEVED]"],
                                      receiver=em_output,
                                      learnable=False),
                    em_output
                    ]
state_to_em_up = [instruct_em,
                  MappingProjection(matrix=up_matrix,
                                    sender=instruct_em.nodes["UP [RETRIEVED]"],
                                    receiver=em_output,
                                    learnable=False),
                  em_output
                  ]
state_to_em_down = [instruct_em,
                    MappingProjection(matrix=down_matrix,
                                      sender=instruct_em.nodes["DOWN [RETRIEVED]"],
                                      receiver=em_output,
                                      learnable=False),
                    em_output
                    ]
state_to_em_open = [instruct_em,
                    MappingProjection(matrix=open_matrix,
                                      sender=instruct_em.nodes["OPEN ACTION [RETRIEVED]"],
                                      receiver=em_output,
                                      learnable=False),
                    em_output
                    ]
state_to_em_pickup = [instruct_em,
                      MappingProjection(matrix=pickup_matrix,
                                        sender=instruct_em.nodes["PICKUP [RETRIEVED]"],
                                        receiver=em_output,
                                        learnable=False),
                      em_output
                      ]
state_to_em_read = [instruct_em,
                    MappingProjection(matrix=read_matrix,
                                      sender=instruct_em.nodes["READ [RETRIEVED]"],
                                      receiver=em_output,
                                      learnable=False),
                    em_output
                    ]

# *********************************************************************************************************************
# **************************************  MLP COMPOSITION  ************************************************************
# *********************************************************************************************************************

hidden_layer = ProcessingMechanism(
    name="HIDDEN",
    default_variable=[0] * 20,
    function=Logistic()
)

mlp_output = ProcessingMechanism(
    name="MLP OUTPUT",
    default_variable=[0, 0, 0, 0, 0, 0, 0],
    function=SoftMax(gain=5.0),
)

goal_to_hidden = MappingProjection(
    sender=state_input,
    receiver=hidden_layer,
    matrix=(0.2 * np.random.rand(obs_len, 20) - 0.1),
    learnable=True
)
hidden_to_output = MappingProjection(
    sender=hidden_layer,
    receiver=mlp_output,
    matrix=(0.2 * np.random.rand(20, 7) - 0.1),
    learnable=True
)

# *********************************************************************************************************************
# ****************************  COMBINED COMPOSITION ********************************************
# *********************************************************************************************************************

combined_comp = AutodiffComposition(
    pathways=[
        # EM pathways (teacher)
        state_to_em_agent_x,
        state_to_em_agent_y,
        state_to_em_door_states,
        state_to_em_key_states,
        state_to_em_holding_key,
        state_to_em_key_color,
        state_to_em_heaven,
        state_to_em_certainty,
        state_to_em_right_value,
        state_to_em_left_value,
        state_to_em_up_value,
        state_to_em_down_value,
        state_to_em_open_value,
        state_to_em_pickup_value,
        state_to_em_read_value,
        state_to_em_right,
        state_to_em_left,
        state_to_em_up,
        state_to_em_down,
        state_to_em_open,
        state_to_em_pickup,
        state_to_em_read,
        # MLP pathway (student)
        [state_input, goal_to_hidden, hidden_layer, hidden_to_output, mlp_output]
    ],
    targets=[(mlp_output, em_output)],
    loss_spec=Loss.MSE,
    learning_rate=0.05,
    name='COMBINED COMPOSITION'
)

# *********************************************************************************************************************
# ******************************************   RUN SIMULATION  ********************************************************
# *********************************************************************************************************************

num_trials = 200


def calculate_entropy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(probs))
    return entropy / max_entropy


def main():
    env = kad.KeysAndDoorsEnv(grid="""
                                    t...
                                    ....
                                    ###.
                                    s...
                                    """)
    reward = 0
    done = False
    print("Running simulation...")
    total_steps = 0
    start_time = timeit.default_timer()

    for trial in range(num_trials):
        observation = env.reset()
        print(f"\n=== Trial {trial + 1}/{num_trials} ===")
        steps = 0

        while True:

            door_input = observation[2] if not isinstance(observation[2], list) else observation[2][0]
            key_input = observation[3] if not isinstance(observation[3], list) else observation[3][0]

            flattened_input = np.array([observation[0], observation[1],
                                        door_input, key_input,
                                        observation[4], observation[5],
                                        observation[6], observation[7]])


            # Now learn with proper target
            combined_comp.learn(
                inputs={state_input: flattened_input},
                execution_mode=ExecutionMode.PyTorch
            )

            # Get EM retrieved values
            right = instruct_em.nodes["RIGHT [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][0]
            left = instruct_em.nodes["LEFT [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][0]
            up = instruct_em.nodes["UP [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][0]
            down = instruct_em.nodes["DOWN [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][0]
            open_action = instruct_em.nodes["OPEN ACTION [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                0]
            pickup_action = instruct_em.nodes["PICKUP [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][0]
            read_action = instruct_em.nodes["READ [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][0]

            right = float(right.flatten()[0])
            left = float(left.flatten()[0])
            up = float(up.flatten()[0])
            down = float(down.flatten()[0])
            open_action = float(open_action.flatten()[0])
            pickup_action = float(pickup_action.flatten()[0])
            read_action = float(read_action.flatten()[0])

            mlp_action = mlp_output.parameters.value.get(combined_comp)

            # Get actions from both teacher and student
            if mlp_action.ndim > 1:
                mlp_action = mlp_action.flatten()

            entropy = calculate_entropy(mlp_action)

            # FIX: Calculate and display loss
            em_action_vec = np.array([right, left, up, down, open_action, pickup_action, read_action])
            loss = np.mean((mlp_action - em_action_vec) ** 2)

            # Execute action in environment
            print(f"Step {steps}: EM={em_action_vec}, MLP={np.round(mlp_action, 3)}, "
                  f"Loss={loss:.4f}, Entropy={entropy:.3f}")

            observation, reward, done = env.step(right, left, up, down, open_action,
                                                 pickup_action, read_action)

            steps += 1
            total_steps += 1

            if RENDER:
                env.render()
            if done:
                print(f"Trial {trial + 1} completed in {steps} steps")
                break

    stop_time = timeit.default_timer()
    print(f'\n{total_steps / (stop_time - start_time):.1f} steps/second, {total_steps} total steps in '
          f'{stop_time - start_time:.2f} seconds')


if RUN:
    if __name__ == "__main__":
        main()
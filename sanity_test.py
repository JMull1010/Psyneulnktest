import timeit
import numpy as np
from psyneulink import *
import KeysAndDoorsWrapper as kad

# Runtime Switches:
RENDER = True
PNL_COMPILE = False
RUN = True

# *****------------------------------------------------------------------------------------------------------------------------
# ****------------------------------------------------------------------------------------------------------------------------  CONSTANTS ***------------------------------------------------------------------------------------------------------------------------
# *****------------------------------------------------------------------------------------------------------------------------

obs_len = 8
num_state_nodes = 8
num_doors = 1
num_keys = 1

# Exploration parameters
EXPLORATION_EPSILON = 0.9  # Start very high — EM is empty so we need lots of random exploration
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.985  # Faster decay so epsilon reaches MIN within ~200 trials
STORE_ON_SUCCESS = True  # Only store successful episodes
REWARD_SHAPING = True  # Store experiences that make progress toward goal
PROGRESS_THRESHOLD = 0.1  # Store if episode gets within this reward of best so far

# *****------------------------------------------------------------------------------------------------------------------------
# ****------------------------------------------------------------------------------------------------------------------------  EM COMPOSITION  *****------------------------------------------------------------------------------------------------------------------------
# *****------------------------------------------------------------------------------------------------------------------------

# State input for EM - expects flat array from wrapper
state_input = ProcessingMechanism(
    name='STATE INPUT',
    default_variable=[0] * 8
)

# NEW: timestep input (not from wrapper) - scalar in [0, 1]
timestep_input = ProcessingMechanism(
    name="TIMESTEP INPUT",
    default_variable=[0]
)
# Action input used only when explicitly storing an experience in EM
action_input = ProcessingMechanism(
    name="ACTION INPUT",
    default_variable=[0, 0, 0, 0, 0, 0, 0]
)
# Output mechanism for EM (teacher signal)
em_output = ProcessingMechanism(
    name="EM OUTPUT",
    default_variable=[0, 0, 0, 0, 0, 0, 0]
)

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

# EM initialization entries — now 16 fields:
# 9 query fields (incl TIME STEP) + 7 action fields
# Initialize with a single template entry (column vector with 16 elements)
em_init_entries = np.zeros((16, 1))  # Single template entry, all zeros initially

# Create EM Composition
instruct_em = EMComposition(
    name="instruct_em",
    memory_template=em_init_entries,
    memory_capacity=300,  # Large enough to store full trajectories
    memory_decay_rate=0,
    memory_fill=0,  # No background noise — slots stay truly empty
    storage_prob=0.0,
    fields={
        "AGENT X": {FIELD_WEIGHT: 2, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        "AGENT Y": {FIELD_WEIGHT: 2, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        "DOOR STATES": {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        "KEY STATES": {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        "HOLDING KEY": {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        "KEY COLOR": {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        "HEAVEN": {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        "CERTAINTY": {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},

        # NEW
        "TIME STEP": {FIELD_WEIGHT: 1, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},

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

# Extraction matrices from wrapper state_input (8 dims) -> each query node (scalar)
x_extract = np.zeros((8, 1));
x_extract[0, 0] = 1
y_extract = np.zeros((8, 1));
y_extract[1, 0] = 1
door_extract = np.zeros((8, 1));
door_extract[2, 0] = 1
key_extract = np.zeros((8, 1));
key_extract[3, 0] = 1
holding_extract = np.zeros((8, 1));
holding_extract[4, 0] = 1
color_extract = np.zeros((8, 1));
color_extract[5, 0] = 1
heaven_extract = np.zeros((8, 1));
heaven_extract[6, 0] = 1
certainty_extract = np.zeros((8, 1));
certainty_extract[7, 0] = 1

# Matrices for mapping EM outputs to action vector
right_matrix = np.array([[1, 0, 0, 0, 0, 0, 0]])
left_matrix = np.array([[0, 1, 0, 0, 0, 0, 0]])
up_matrix = np.array([[0, 0, 1, 0, 0, 0, 0]])
down_matrix = np.array([[0, 0, 0, 1, 0, 0, 0]])
open_matrix = np.array([[0, 0, 0, 0, 1, 0, 0]])
pickup_matrix = np.array([[0, 0, 0, 0, 0, 1, 0]])
read_matrix = np.array([[0, 0, 0, 0, 0, 0, 1]])

# Query pathways (state_input -> EM query nodes)
state_to_em_agent_x = [state_input,
                       MappingProjection(matrix=x_extract, sender=state_input,
                                         receiver=instruct_em.nodes["AGENT X [QUERY]"], learnable=False),
                       instruct_em]
state_to_em_agent_y = [state_input,
                       MappingProjection(matrix=y_extract, sender=state_input,
                                         receiver=instruct_em.nodes["AGENT Y [QUERY]"], learnable=False),
                       instruct_em]
state_to_em_door_states = [state_input,
                           MappingProjection(matrix=door_extract, sender=state_input,
                                             receiver=instruct_em.nodes["DOOR STATES [QUERY]"], learnable=False),
                           instruct_em]
state_to_em_key_states = [state_input,
                          MappingProjection(matrix=key_extract, sender=state_input,
                                            receiver=instruct_em.nodes["KEY STATES [QUERY]"], learnable=False),
                          instruct_em]
state_to_em_holding_key = [state_input,
                           MappingProjection(matrix=holding_extract, sender=state_input,
                                             receiver=instruct_em.nodes["HOLDING KEY [QUERY]"], learnable=False),
                           instruct_em]
state_to_em_key_color = [state_input,
                         MappingProjection(matrix=color_extract, sender=state_input,
                                           receiver=instruct_em.nodes["KEY COLOR [QUERY]"], learnable=False),
                         instruct_em]
state_to_em_heaven = [state_input,
                      MappingProjection(matrix=heaven_extract, sender=state_input,
                                        receiver=instruct_em.nodes["HEAVEN [QUERY]"], learnable=False),
                      instruct_em]
state_to_em_certainty = [state_input,
                         MappingProjection(matrix=certainty_extract, sender=state_input,
                                           receiver=instruct_em.nodes["CERTAINTY [QUERY]"], learnable=False),
                         instruct_em]

# NEW: timestep_input -> EM "TIME STEP [QUERY]"  (already normalized before feeding)
timestep_to_em_time_step = [timestep_input,
                            MappingProjection(matrix=np.array([[1.0]]), sender=timestep_input,
                                              receiver=instruct_em.nodes["TIME STEP [QUERY]"], learnable=False),
                            instruct_em]

# Action pathways into EM target/value fields (used only during explicit storage)
action_right_extract = np.zeros((7, 1)); action_right_extract[0, 0] = 1
action_left_extract = np.zeros((7, 1)); action_left_extract[1, 0] = 1
action_up_extract = np.zeros((7, 1)); action_up_extract[2, 0] = 1
action_down_extract = np.zeros((7, 1)); action_down_extract[3, 0] = 1
action_open_extract = np.zeros((7, 1)); action_open_extract[4, 0] = 1
action_pickup_extract = np.zeros((7, 1)); action_pickup_extract[5, 0] = 1
action_read_extract = np.zeros((7, 1)); action_read_extract[6, 0] = 1

action_to_em_right_value = [action_input,
                            MappingProjection(matrix=action_right_extract, sender=action_input,
                                              receiver=instruct_em.nodes["RIGHT [VALUE]"], learnable=False),
                          instruct_em]

action_to_em_left_value = [action_input,
                           MappingProjection(matrix=action_left_extract, sender=action_input,
                                             receiver=instruct_em.nodes["LEFT [VALUE]"], learnable=False),
                           instruct_em]
action_to_em_up_value = [action_input,
                         MappingProjection(matrix=action_up_extract, sender=action_input,
                                           receiver=instruct_em.nodes["UP [VALUE]"], learnable=False),
                         instruct_em]
action_to_em_down_value = [action_input,
                           MappingProjection(matrix=action_down_extract, sender=action_input,
                                             receiver=instruct_em.nodes["DOWN [VALUE]"], learnable=False),
                           instruct_em]
action_to_em_open_value = [action_input,
                           MappingProjection(matrix=action_open_extract, sender=action_input,
                                             receiver=instruct_em.nodes["OPEN ACTION [VALUE]"], learnable=False),
                           instruct_em]
action_to_em_pickup_value = [action_input,
                             MappingProjection(matrix=action_pickup_extract, sender=action_input,
                                               receiver=instruct_em.nodes["PICKUP [VALUE]"], learnable=False),
                             instruct_em]
action_to_em_read_value = [action_input,
                           MappingProjection(matrix=action_read_extract, sender=action_input,
                                             receiver=instruct_em.nodes["READ [VALUE]"], learnable=False),
                           instruct_em]

# Map retrieved action fields into em_output vector (7 dims)
state_to_em_right = [instruct_em,
                     MappingProjection(matrix=right_matrix, sender=instruct_em.nodes["RIGHT [RETRIEVED]"],
                                       receiver=em_output, learnable=False),
                     em_output]
state_to_em_left = [instruct_em,
                    MappingProjection(matrix=left_matrix, sender=instruct_em.nodes["LEFT [RETRIEVED]"],
                                      receiver=em_output, learnable=False),
                    em_output]
state_to_em_up = [instruct_em,
                  MappingProjection(matrix=up_matrix, sender=instruct_em.nodes["UP [RETRIEVED]"],
                                    receiver=em_output, learnable=False),
                  em_output]
state_to_em_down = [instruct_em,
                    MappingProjection(matrix=down_matrix, sender=instruct_em.nodes["DOWN [RETRIEVED]"],
                                      receiver=em_output, learnable=False),
                    em_output]
state_to_em_open = [instruct_em,
                    MappingProjection(matrix=open_matrix, sender=instruct_em.nodes["OPEN ACTION [RETRIEVED]"],
                                      receiver=em_output, learnable=False),
                    em_output]
state_to_em_pickup = [instruct_em,
                      MappingProjection(matrix=pickup_matrix, sender=instruct_em.nodes["PICKUP [RETRIEVED]"],
                                        receiver=em_output, learnable=False),
                      em_output]
state_to_em_read = [instruct_em,
                    MappingProjection(matrix=read_matrix, sender=instruct_em.nodes["READ [RETRIEVED]"],
                                      receiver=em_output, learnable=False),
                    em_output]

# *****------------------------------------------------------------------------------------------------------------------------
# ****------------------------------------------------------------------------------------------------------------------------  MLP COMPOSITION  ****------------------------------------------------------------------------------------------------------------------------
# *****------------------------------------------------------------------------------------------------------------------------

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
    matrix=(0.2 * np.random.rand(obs_len, 20) - 0.1),  # Range: [-0.1, 0.1]
    learnable=True
)
hidden_to_output = MappingProjection(
    sender=hidden_layer,
    receiver=mlp_output,
    matrix=(0.2 * np.random.rand(20, 7) - 0.1),  # Range: [-0.1, 0.1]
    learnable=True
)

# *****------------------------------------------------------------------------------------------------------------------------
# ****------------------------------------------------------------------------------------------------------------------------  COMBINED COMPOSITION ****------------------------------------------------------------------------------------------------------------------------
# *****------------------------------------------------------------------------------------------------------------------------

combined_comp = AutodiffComposition(
    pathways=[
        # EM pathways
        state_to_em_agent_x,
        state_to_em_agent_y,
        state_to_em_door_states,
        state_to_em_key_states,
        state_to_em_holding_key,
        state_to_em_key_color,
        state_to_em_heaven,
        state_to_em_certainty,

        # NEW
        timestep_to_em_time_step,

        action_to_em_right_value,
        action_to_em_left_value,
        action_to_em_up_value,
        action_to_em_down_value,
        action_to_em_open_value,
        action_to_em_pickup_value,
        action_to_em_read_value,
        state_to_em_right,
        state_to_em_left,
        state_to_em_up,
        state_to_em_down,
        state_to_em_open,
        state_to_em_pickup,
        state_to_em_read,

        # MLP pathway (learner/logging)
        [state_input, goal_to_hidden, hidden_layer, hidden_to_output, mlp_output]
    ],
    targets=[(mlp_output, em_output)],  # MLP learns from EM's retrieved action
    loss_spec=Loss.MSE,
    learning_rate=0.05,
    name='COMBINED COMPOSITION'
)

# *****------------------------------------------------------------------------------------------------------------------------
# ****------------------------------------------------------------------------------------------------------------------------  RUN SIMULATION  ****------------------------------------------------------------------------------------------------------------------------
# *****------------------------------------------------------------------------------------------------------------------------

num_trials = 200
MAX_STEPS_PER_EPISODE = 200  # Force reset if stuck

NEXT_EM_SLOT = 0
EM_CAPACITY = 300  # Match the instruct_em memory_capacity

EM_SLOT_REWARD = np.full(EM_CAPACITY, -np.inf, dtype=float)
EM_SLOT_STEPS = np.full(EM_CAPACITY, np.inf, dtype=float)


def calculate_entropy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(probs))
    return entropy / max_entropy


def get_em_action():
    """Extract the EM's retrieved action and convert to a one-hot vector."""
    right_val = float(
        instruct_em.nodes["RIGHT [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][0].flatten()[0])
    left_val = float(
        instruct_em.nodes["LEFT [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][0].flatten()[0])
    up_val = float(instruct_em.nodes["UP [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][0].flatten()[0])
    down_val = float(
        instruct_em.nodes["DOWN [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][0].flatten()[0])
    open_val = float(
        instruct_em.nodes["OPEN ACTION [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][0].flatten()[0])
    pickup_val = float(
        instruct_em.nodes["PICKUP [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][0].flatten()[0])
    read_val = float(
        instruct_em.nodes["READ [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][0].flatten()[0])

    em_action_raw = np.array([right_val, left_val, up_val, down_val, open_val, pickup_val, read_val])

    action_vec = np.zeros(7)
    action_vec[np.argmax(em_action_raw)] = 1.0
    return action_vec


def episode_better(new_reward, new_steps, old_reward, old_steps):
    if new_reward > old_reward:
        return True
    if np.isclose(new_reward, old_reward) and new_steps < old_steps:
        return True
    return False


def store_experience_in_em(state, step_number, action_vec, episode_reward, episode_steps):
    """
    Store (state, relative_timestep, action) into EM by running it in learn mode.
    """
    global NEXT_EM_SLOT, EM_SLOT_REWARD, EM_SLOT_STEPS

    relative_timestep = step_number / MAX_STEPS_PER_EPISODE
    timestep_arr = np.array([relative_timestep], dtype=float)

    slot = NEXT_EM_SLOT % EM_CAPACITY
    slot_empty = np.isneginf(EM_SLOT_REWARD[slot])
    should_overwrite = slot_empty or episode_better(
        episode_reward, episode_steps,
        EM_SLOT_REWARD[slot], EM_SLOT_STEPS[slot]
    )

    if should_overwrite:
        # Store by running the composition in learn mode
        comp_inputs = {
            state_input: state,
            timestep_input: timestep_arr,
            action_input: action_vec,
        }

        # Drive EM input nodes through the composition, then explicitly encode one memory.
        # NOTE: in nested Autodiff/PyTorch execution, runtime storage_prob context can differ
        # from `combined_comp`, so we explicitly call _encode_memory here for deterministic storage.
        combined_comp.run(
            inputs=comp_inputs,
            execution_mode=ExecutionMode.PyTorch
        )
        instruct_em._encode_memory(context=combined_comp)
        EM_SLOT_REWARD[slot] = float(episode_reward)
        EM_SLOT_STEPS[slot] = float(episode_steps)

        if slot_empty:
            print(
                f"  → Stored in EM slot {slot} (step {step_number}, rel_t={relative_timestep:.4f}, reward={episode_reward:.2f})")
        else:
            print(
                f"  → Replaced EM slot {slot} (step {step_number}, rel_t={relative_timestep:.4f}, better: {episode_reward:.2f}@{episode_steps})")
    else:
        print(f"  → Skipped EM slot {slot} (episode not better)")

    NEXT_EM_SLOT += 1


def main():
    env = kad.KeysAndDoorsEnv(grid="""
                                    t...
                                    ....
                                    ###.
                                    s...
                                    """)

    epsilon = EXPLORATION_EPSILON
    print("Running MLP-based exploration followed by pure EM...")
    print(f"Starting epsilon: {epsilon:.3f} (MLP used for exploration; entropy is logged only)")
    total_steps = 0
    start_time = timeit.default_timer()

    successful_episodes = []
    best_reward = -float('inf')

    for trial in range(num_trials):
        observation = env.reset()
        filled_slots = int(np.sum(~np.isneginf(EM_SLOT_REWARD)))

        if len(successful_episodes) == 0:
            print(
                f"\n=== Trial {trial + 1}/{num_trials} (MODE: MLP Exploration, EM slots filled={filled_slots}/{EM_CAPACITY}) ===")
        else:
            print(
                f"\n=== Trial {trial + 1}/{num_trials} (MODE: Pure EM, EM slots filled={filled_slots}/{EM_CAPACITY}) ===")

        steps = 0
        episode_reward = 0
        episode_trajectory = []

        while True:
            door_input = observation[2] if not isinstance(observation[2], list) else observation[2][0]
            key_input = observation[3] if not isinstance(observation[3], list) else observation[3][0]

            flattened_input = np.array([observation[0], observation[1],
                                        door_input, key_input,
                                        observation[4], observation[5],
                                        observation[6], observation[7]])

            # NORMALIZED timestep in [0,1] based on current step / MAX
            step_number = steps + 1  # Steps is 0-indexed, so step_number is 1, 2, 3, ...
            timestep_norm = step_number / MAX_STEPS_PER_EPISODE

            # Inputs to composition (state + normalized timestep)
            comp_inputs = {
                state_input: flattened_input,
                timestep_input: np.array([timestep_norm], dtype=float),
                action_input: np.zeros(7, dtype=float)
            }

            # Choose action: MLP (before first success) else EM (after first success)
            if len(successful_episodes) == 0:
                # Before first success: use MLP for exploration
                exploration_type = "MLP"

                combined_comp.run(
                    inputs=comp_inputs,
                    execution_mode=ExecutionMode.PyTorch
                )

                # Get action from MLP by sampling from softmax distribution
                mlp_action = mlp_output.parameters.value.get(combined_comp)
                mlp_action = np.array(mlp_action).flatten()
                # Sample action based on softmax probabilities
                action_idx = np.random.choice(7, p=mlp_action)
                action_vec = np.zeros(7)
                action_vec[action_idx] = 1.0
            else:
                # After first success: only use EM
                exploration_type = "EM"

                combined_comp.learn(
                    inputs=comp_inputs,
                    execution_mode=ExecutionMode.PyTorch
                )

                print("\n  ━━━ EM RETRIEVED MEMORY ━━━")
                retrieved_fields = {
                    "AGENT X": instruct_em.nodes["AGENT X [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                        0].flatten()[0],
                    "AGENT Y": instruct_em.nodes["AGENT Y [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                        0].flatten()[0],
                    "DOOR STATES":
                        instruct_em.nodes["DOOR STATES [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                            0].flatten()[0],
                    "KEY STATES":
                        instruct_em.nodes["KEY STATES [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                            0].flatten()[0],
                    "HOLDING KEY":
                        instruct_em.nodes["HOLDING KEY [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                            0].flatten()[0],
                    "KEY COLOR":
                        instruct_em.nodes["KEY COLOR [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                            0].flatten()[0],
                    "HEAVEN": instruct_em.nodes["HEAVEN [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                        0].flatten()[0],
                    "CERTAINTY":
                        instruct_em.nodes["CERTAINTY [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                            0].flatten()[0],
                    "TIME STEP":
                        instruct_em.nodes["TIME STEP [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                            0].flatten()[0],
                    "RIGHT": instruct_em.nodes["RIGHT [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                        0].flatten()[0],
                    "LEFT": instruct_em.nodes["LEFT [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                        0].flatten()[0],
                    "UP": instruct_em.nodes["UP [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                        0].flatten()[0],
                    "DOWN": instruct_em.nodes["DOWN [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                        0].flatten()[0],
                    "OPEN ACTION":
                        instruct_em.nodes["OPEN ACTION [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                            0].flatten()[0],
                    "PICKUP": instruct_em.nodes["PICKUP [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                        0].flatten()[0],
                    "READ": instruct_em.nodes["READ [RETRIEVED]"].parameters.value.get('COMBINED COMPOSITION')[0][
                        0].flatten()[0],
                }
                for field_name, value in retrieved_fields.items():
                    print(f"    {field_name:15s}: {value:.4f}")
                print("  ━━━━━━━━━━━━━━━━━━━━━━━━━\n")
                # ═══════════════════════════════════════════════════════

                action_vec = get_em_action()


            if len(successful_episodes) == 1:
                print("\n" + "=" * 70)
                print("  FIRST SUCCESS ACHIEVED! Switching to PURE EM mode...")

                # DEBUG: Print what's actually stored in EM
                print("\n  DEBUG: Checking stored memories...")
                current_memory = instruct_em.parameters.memory.get(combined_comp)
                for i in range(min(5, len(current_memory))):  # Print first 5 memories
                    mem = current_memory[i].flatten()
                    print(
                        f"  Slot {i}: timestep={mem[8]:.4f}, action=[{mem[9]:.0f},{mem[10]:.0f},{mem[11]:.0f},{mem[12]:.0f},{mem[13]:.0f},{mem[14]:.0f},{mem[15]:.0f}]")

                print("=" * 70 + "\n")

            # Log MLP entropy (MLP is only used for action before first success)
            mlp_action = mlp_output.parameters.value.get(combined_comp)
            mlp_action = np.array(mlp_action).flatten()
            entropy = calculate_entropy(mlp_action)

            print(f"Step {steps}: Action={exploration_type}, "
                  f"step_num={step_number}, t_norm={timestep_norm:.4f}, "
                  f"MLP={np.round(mlp_action, 3)}, Entropy={entropy:.3f}")

            # Store state, step_number, action for later EM storage
            episode_trajectory.append((flattened_input.copy(), step_number, action_vec.copy()))

            observation, reward, done = env.step(
                action_vec[0], action_vec[1], action_vec[2], action_vec[3],
                action_vec[4], action_vec[5], action_vec[6]
            )

            episode_reward += reward
            steps += 1
            total_steps += 1

            if RENDER:
                env.render()

            if steps >= MAX_STEPS_PER_EPISODE:
                print(f"  TIMEOUT after {steps} steps - moving to next trial")
                break

            if done:
                print(f"Trial {trial + 1} completed in {steps} steps (reward={episode_reward:.2f})")

                should_store = False
                if (STORE_ON_SUCCESS and episode_reward > 0) and len(successful_episodes) == 0:
                    should_store = True
                    print(f"  ✓ SUCCESS!")
                elif REWARD_SHAPING and episode_reward > best_reward - PROGRESS_THRESHOLD and len(successful_episodes) == 0:
                    should_store = True
                    print(f"  ↗ PROGRESS! (best so far: {best_reward:.2f})")

                if should_store:
                    print(f"  Attempting to store {len(episode_trajectory)} experiences in EM "
                          f"(episode reward={episode_reward:.2f}, steps={steps})")
                    for state, step_num, action in episode_trajectory:
                        store_experience_in_em(state, step_num, action, episode_reward, steps)

                    successful_episodes.append(trial)
                    best_reward = max(best_reward, episode_reward)

                    if len(successful_episodes) == 1:
                        print("\n" + "=" * 70)
                        print("  FIRST SUCCESS ACHIEVED! Switching to PURE EM mode...")
                        print("  All future trials will use EM retrieval only (sanity check)")
                        print("=" * 70 + "\n")

                break

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        filled_slots = int(np.sum(~np.isneginf(EM_SLOT_REWARD)))

    stop_time = timeit.default_timer()

    print(f'\n=== SUMMARY ===')
    print(f'Successful episodes: {len(successful_episodes)}/{num_trials}')
    print(f'EM slots filled: {int(np.sum(~np.isneginf(EM_SLOT_REWARD)))}/{EM_CAPACITY}')
    print(f'{total_steps / (stop_time - start_time):.1f} steps/second')
    print(f'{total_steps} total steps in {stop_time - start_time:.2f} seconds')


if RUN:
    if __name__ == "__main__":
        main()
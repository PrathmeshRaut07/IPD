# RL.py
import random

def compute_reward(metrics):
    """
    Compute an overall reward based on the audio analysis metrics.
    
    - clarity_score: Higher is better.
    - pitch_rate: The closer to the ideal (150 Hz), the better.
    - tone & emotion: Mapped to numerical rewards.
    """
    clarity = metrics.get("clarity_score", 0.0)
    pitch = metrics.get("pitch_rate", 150)
    tone = metrics.get("tone", "Neutral")
    emotion = metrics.get("emotion", "Neutral")
    
    # --- Pitch Reward ---
    ideal_pitch = 150
    pitch_diff = abs(pitch - ideal_pitch)
    pitch_reward = max(0.0, 1 - (pitch_diff / ideal_pitch))
    
    # --- Tone Reward ---
    tone_rewards = {
        "Confident": 1.0,
        "Formal": 0.8,
        "Persuasive": 0.7,
        "Neutral": 0.5,
        "Casual": 0.3
    }
    tone_reward = tone_rewards.get(tone, 0.5)
    
    # --- Emotion Reward ---
    emotion_rewards = {
        "Calm": 1.0,
        "Happy": 0.9,
        "Excited": 0.8,
        "Neutral": 0.5,
        "Sad": 0.3,
        "Angry": 0.1
    }
    emotion_reward = emotion_rewards.get(emotion, 0.5)
    
    # --- Weighted Sum ---
    w_clarity = 0.4
    w_pitch = 0.2
    w_tone = 0.2
    w_emotion = 0.2
    
    overall_reward = (w_clarity * clarity +
                      w_pitch * pitch_reward +
                      w_tone * tone_reward +
                      w_emotion * emotion_reward)
    return overall_reward

def policy_recommendation(metrics):
    """
    Provides human-readable recommendations based on the metrics.
    """
    recommendations = []
    clarity = metrics.get("clarity_score", 0.0)
    pitch = metrics.get("pitch_rate", 150)
    tone = metrics.get("tone", "Neutral")
    emotion = metrics.get("emotion", "Neutral")
    
    # Recommendation based on clarity
    if clarity < 0.7:
        recommendations.append("Speak slower and enunciate your words more clearly.")
    
    # Recommendation based on pitch (assume ideal=150 Hz with ±20 Hz tolerance)
    ideal_pitch = 150
    if pitch < ideal_pitch - 20:
        recommendations.append("Try to raise your pitch to sound more energetic.")
    elif pitch > ideal_pitch + 20:
        recommendations.append("Lower your pitch for a calmer delivery.")
    
    # Recommendation based on tone
    if tone == "Casual":
        recommendations.append("Adopt a more confident tone to enhance professionalism.")
    elif tone == "Neutral":
        recommendations.append("Infuse more conviction into your tone.")
    
    # Recommendation based on emotion
    if emotion in ["Sad", "Angry"]:
        recommendations.append("Maintain a calm and positive demeanor while speaking.")
    
    if not recommendations:
        recommendations.append("Your speech delivery is excellent. Keep it up!")
    
    return " ".join(recommendations)

class RLAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        """
        Initialize the RL agent.
          - actions: A list of possible actions (recommendations).
          - alpha: Learning rate.
          - gamma: Discount factor.
          - epsilon: Exploration rate.
        """
        self.actions = actions  
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # Q-values keyed by state tuples

    def get_state(self, metrics):
        """
        Convert continuous audio metrics into a discrete state.
        State is defined as a tuple: (clarity_level, pitch_level, tone_level, emotion_level)
        """
        clarity = metrics.get("clarity_score", 0.0)
        pitch = metrics.get("pitch_rate", 150)
        tone = metrics.get("tone", "Neutral")
        emotion = metrics.get("emotion", "Neutral")
        
        # Discretize clarity: 0=low, 1=medium, 2=high
        if clarity < 0.4:
            clarity_level = 0
        elif clarity < 0.7:
            clarity_level = 1
        else:
            clarity_level = 2
        
        # Discretize pitch: 0=low, 1=ideal, 2=high (using 130 and 170 Hz as thresholds)
        if pitch < 130:
            pitch_level = 0
        elif pitch > 170:
            pitch_level = 2
        else:
            pitch_level = 1
        
        # Map tone to an integer (defaulting to Neutral=1)
        tone_map = {"Casual": 0, "Neutral": 1, "Persuasive": 2, "Formal": 3, "Confident": 4}
        tone_level = tone_map.get(tone, 1)
        
        # Map emotion to an integer (defaulting to Neutral=1)
        emotion_map = {"Angry": 0, "Sad": 1, "Excited": 2, "Happy": 3, "Calm": 4, "Neutral": 1}
        emotion_level = emotion_map.get(emotion, 1)
        
        return (clarity_level, pitch_level, tone_level, emotion_level)
    
    def choose_action(self, state):
        """
        Choose an action using an epsilon-greedy policy.
        """
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            # Choose the action with the highest Q-value in the current state.
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def update(self, state, action, reward, next_state):
        """
        Update Q-values based on the observed transition.
        """
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        # Q-learning update rule.
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def act(self, current_metrics, next_metrics):
        """
        Given the current and next metrics, choose an action, compute reward,
        and update Q-values.
        Returns the chosen action and the computed reward.
        """
        state = self.get_state(current_metrics)
        next_state = self.get_state(next_metrics)
        action = self.choose_action(state)
        reward = compute_reward(current_metrics)
        self.update(state, action, reward, next_state)
        return action, reward

# Expanded list of possible actions
actions = [
    "NoAction",
    "ImproveClarity",
    "RaisePitch",
    "LowerPitch",
    "AdoptConfidentTone",
    "MaintainCalm",
    "SpeakSlower",
    "SpeakFaster",
    "EnunciateBetter",
    "IncreaseVolume",
    "DecreaseVolume"
]

# Instantiate a global RL agent that can be imported elsewhere.
global_rl_agent = RLAgent(actions, alpha=0.1, gamma=0.9, epsilon=0.2)

def get_rl_agent():
    """Return the global RL agent instance."""
    return global_rl_agent

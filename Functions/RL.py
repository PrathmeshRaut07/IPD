# import random

# def compute_reward(metrics):
#     """
#     Compute an overall reward based on the audio analysis metrics.
    
#     - clarity_score: Higher is better.
#     - pitch_rate: The closer to the ideal (150 Hz), the better.
#     - tone & emotion: Mapped to numerical rewards.
#     """
#     clarity = metrics.get("clarity_score", 0.0)
#     pitch = metrics.get("pitch_rate", 150)
#     tone = metrics.get("tone", "Neutral")
#     emotion = metrics.get("emotion", "Neutral")
    
  
#     ideal_pitch = 150
#     pitch_diff = abs(pitch - ideal_pitch)
#     pitch_reward = max(0.0, 1 - (pitch_diff / ideal_pitch))
    

#     tone_rewards = {
#         "Confident": 1.0,
#         "Formal": 0.8,
#         "Persuasive": 0.7,
#         "Neutral": 0.5,
#         "Casual": 0.3
#     }
#     tone_reward = tone_rewards.get(tone, 0.5)

#     emotion_rewards = {
#         "Calm": 1.0,
#         "Happy": 0.9,
#         "Excited": 0.8,
#         "Neutral": 0.5,
#         "Sad": 0.3,
#         "Angry": 0.1
#     }
#     emotion_reward = emotion_rewards.get(emotion, 0.5)

#     w_clarity = 0.4
#     w_pitch = 0.2
#     w_tone = 0.2
#     w_emotion = 0.2
    
#     overall_reward = (w_clarity * clarity +
#                       w_pitch * pitch_reward +
#                       w_tone * tone_reward +
#                       w_emotion * emotion_reward)
#     return overall_reward

# def policy_recommendation(metrics):
#     """
#     Provides human-readable recommendations based on the metrics.
#     """
#     recommendations = []
#     clarity = metrics.get("clarity_score", 0.0)
#     pitch = metrics.get("pitch_rate", 150)
#     tone = metrics.get("tone", "Neutral")
#     emotion = metrics.get("emotion", "Neutral")
    
#     # Recommendation based on clarity
#     if clarity < 0.7:
#         recommendations.append("Speak slower and enunciate your words more clearly.")
    

#     ideal_pitch = 150
#     if pitch < ideal_pitch - 20:
#         recommendations.append("Try to raise your pitch to sound more energetic.")
#     elif pitch > ideal_pitch + 20:
#         recommendations.append("Lower your pitch for a calmer delivery.")
    
    
#     if tone == "Casual":
#         recommendations.append("Adopt a more confident tone to enhance professionalism.")
#     elif tone == "Neutral":
#         recommendations.append("Infuse more conviction into your tone.")
    
    
#     if emotion in ["Sad", "Angry"]:
#         recommendations.append("Maintain a calm and positive demeanor while speaking.")
    
#     if not recommendations:
#         recommendations.append("Your speech delivery is excellent. Keep it up!")
    
#     return " ".join(recommendations)

# class RLAgent:
#     def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
#         """
#         Initialize the RL agent.
#           - actions: A list of possible actions (recommendations).
#           - alpha: Learning rate.
#           - gamma: Discount factor.
#           - epsilon: Exploration rate.
#         """
#         self.actions = actions  
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.q_table = {}  # Q-values keyed by state tuples

#     def get_state(self, metrics):
#         """
#         Convert continuous audio metrics into a discrete state.
#         State is defined as a tuple: (clarity_level, pitch_level, tone_level, emotion_level)
#         """
#         clarity = metrics.get("clarity_score", 0.0)
#         pitch = metrics.get("pitch_rate", 150)
#         tone = metrics.get("tone", "Neutral")
#         emotion = metrics.get("emotion", "Neutral")
    
#         if clarity < 0.4:
#             clarity_level = 0
#         elif clarity < 0.7:
#             clarity_level = 1
#         else:
#             clarity_level = 2
        
#         if pitch < 130:
#             pitch_level = 0
#         elif pitch > 170:
#             pitch_level = 2
#         else:
#             pitch_level = 1
        
#         # Map tone to an integer (defaulting to Neutral=1)
#         tone_map = {"Casual": 0, "Neutral": 1, "Persuasive": 2, "Formal": 3, "Confident": 4}
#         tone_level = tone_map.get(tone, 1)
        
#         # Map emotion to an integer (defaulting to Neutral=1)
#         emotion_map = {"Angry": 0, "Sad": 1, "Excited": 2, "Happy": 3, "Calm": 4, "Neutral": 1}
#         emotion_level = emotion_map.get(emotion, 1)
        
#         return (clarity_level, pitch_level, tone_level, emotion_level)
    
#     def choose_action(self, state):
#         """
#         Choose an action using an epsilon-greedy policy.
#         """
#         if state not in self.q_table:
#             self.q_table[state] = {action: 0.0 for action in self.actions}
#         if random.random() < self.epsilon:
#             return random.choice(self.actions)
#         else:
#             # Choose the action with the highest Q-value in the current state.
#             return max(self.q_table[state], key=self.q_table[state].get)
    
#     def update(self, state, action, reward, next_state):
#         """
#         Update Q-values based on the observed transition.
#         """
#         if state not in self.q_table:
#             self.q_table[state] = {a: 0.0 for a in self.actions}
#         if next_state not in self.q_table:
#             self.q_table[next_state] = {a: 0.0 for a in self.actions}
        
#         current_q = self.q_table[state][action]
#         max_next_q = max(self.q_table[next_state].values())
#         # Q-learning update rule.
#         new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
#         self.q_table[state][action] = new_q
    
#     def act(self, current_metrics, next_metrics):
#         """
#         Given the current and next metrics, choose an action, compute reward,
#         and update Q-values.
#         Returns the chosen action and the computed reward.
#         """
#         state = self.get_state(current_metrics)
#         next_state = self.get_state(next_metrics)
#         action = self.choose_action(state)
#         reward = compute_reward(current_metrics)
#         self.update(state, action, reward, next_state)
#         return action, reward


# actions = [
#     "NoAction",
#     "ImproveClarity",
#     "RaisePitch",
#     "LowerPitch",
#     "AdoptConfidentTone",
#     "MaintainCalm",
#     "SpeakSlower",
#     "SpeakFaster",
#     "EnunciateBetter",
#     "IncreaseVolume",
#     "DecreaseVolume"
# ]


# global_rl_agent = RLAgent(actions, alpha=0.1, gamma=0.9, epsilon=0.2)

# def get_rl_agent():
#     """Return the global RL agent instance."""
#     return global_rl_agent


import random
import numpy as np
from collections import defaultdict

def compute_reward(metrics):
    """
    Enhanced reward computation with weighted multi-objective optimization.
    """
    clarity = metrics.get("clarity_score", 0.0)
    pitch = metrics.get("pitch_rate", 150)
    tone = metrics.get("tone", "Neutral")
    emotion = metrics.get("emotion", "Neutral")
    
    clarity_reward = np.tanh(2 * clarity)
    

    ideal_pitch = 150
    pitch_variance = 30
    pitch_reward = np.exp(-((pitch - ideal_pitch) ** 2) / (2 * pitch_variance ** 2))
  
    tone_rewards = {
        "Confident": 1.0,
        "Formal": 0.85,
        "Persuasive": 0.8,
        "Neutral": 0.6,
        "Casual": 0.4
    }
    tone_reward = tone_rewards.get(tone, 0.5)

    emotion_rewards = {
        "Calm": 1.0,
        "Happy": 0.9,
        "Excited": 0.8,
        "Neutral": 0.6,
        "Sad": 0.3,
        "Angry": 0.1
    }
    emotion_reward = emotion_rewards.get(emotion, 0.5)
    

    w_clarity = 0.35
    w_pitch = 0.25
    w_tone = 0.2
    w_emotion = 0.2
  
    overall_reward = (w_clarity * clarity_reward +
                     w_pitch * pitch_reward +
                     w_tone * tone_reward +
                     w_emotion * emotion_reward)
 
    shaped_reward = np.tanh(1.5 * overall_reward)
    
    return shaped_reward

def policy_recommendation(metrics):
    """
    Enhanced recommendation system with prioritized, context-aware suggestions.
    """
    recommendations = []
    clarity = metrics.get("clarity_score", 0.0)
    pitch = metrics.get("pitch_rate", 150)
    tone = metrics.get("tone", "Neutral")
    emotion = metrics.get("emotion", "Neutral")
 
    if clarity < 0.5:
        recommendations.append("Critical: Focus on clear enunciation and slower speech rate.")
    elif clarity < 0.7:
        recommendations.append("Important: Enhance word clarity and maintain a steady pace.")
    
    ideal_pitch = 150
    pitch_diff = abs(pitch - ideal_pitch)
    if pitch_diff > 30:
        if pitch < ideal_pitch:
            recommendations.append("Gradually raise your pitch for better engagement.")
        else:
            recommendations.append("Slightly lower your pitch for optimal delivery.")

    if tone in ["Casual", "Neutral"]:
        if clarity > 0.6: 
            recommendations.append("Project more confidence while maintaining your clear delivery.")

    if emotion in ["Sad", "Angry"]:
        recommendations.append("Maintain professional composure while expressing your points.")
    elif emotion == "Excited" and clarity < 0.6:
        recommendations.append("Channel your enthusiasm while focusing on clear articulation.")
    
    if not recommendations:
        recommendations.append("Excellent delivery! Consider fine-tuning pitch modulation for even better impact.")
    
    return " ".join(recommendations)

class RLAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=0.15):
        """
        Enhanced RL agent with advanced exploration and learning capabilities.
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Initialize Q-table with default values for all actions
        self.q_table = defaultdict(lambda: {action: 0.0 for action in actions})
        self.visit_counts = defaultdict(lambda: defaultdict(int))
        self.total_updates = 0
        
    def get_state(self, metrics):
        """
        Enhanced state representation with continuous-to-discrete mapping.
        """
        clarity = metrics.get("clarity_score", 0.0)
        pitch = metrics.get("pitch_rate", 150)
        tone = metrics.get("tone", "Neutral")
        emotion = metrics.get("emotion", "Neutral")
      
        clarity_level = int(np.floor(clarity * 5))  # 0-4 levels
        

        pitch_ranges = [120, 140, 160, 180]
        pitch_level = sum(pitch > p for p in pitch_ranges)
        
        tone_map = {
            "Casual": 0, "Neutral": 1, "Persuasive": 2,
            "Formal": 3, "Confident": 4
        }
        tone_level = tone_map.get(tone, 1)
        
        # Context-aware emotion mapping
        emotion_map = {
            "Angry": 0, "Sad": 1, "Neutral": 2,
            "Excited": 3, "Happy": 4, "Calm": 5
        }
        emotion_level = emotion_map.get(emotion, 2)
        
        return (clarity_level, pitch_level, tone_level, emotion_level)
    
    def choose_action(self, state):
        """
        Enhanced action selection with UCB exploration.
        """
        # Ensure state exists in Q-table
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        
        if random.random() < self.epsilon:
            action_visits = {a: self.visit_counts[state][a] for a in self.actions}
            min_visits = min(action_visits.values())
            least_visited = [a for a, v in action_visits.items() if v == min_visits]
            return random.choice(least_visited)
        
        C = 2.0  
        N = sum(self.visit_counts[state].values()) + 1
        
        def ucb_value(action):
            Q = self.q_table[state][action]
            n = self.visit_counts[state][action] + 1
            return Q + C * np.sqrt(np.log(N) / n)
        
        return max(self.actions, key=ucb_value)
    
    def update(self, state, action, reward, next_state):
        """
        Enhanced Q-learning update with experience replay and adaptive learning.
        """
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in self.actions}

        self.visit_counts[state][action] += 1
        self.total_updates += 1
        

        effective_alpha = self.alpha / np.sqrt(self.visit_counts[state][action])
        

        next_action = max(self.q_table[next_state].items(), key=lambda x: x[1])[0]
        next_q = self.q_table[next_state][next_action]
        

        current_q = self.q_table[state][action]
        new_q = current_q + effective_alpha * (
            reward + self.gamma * next_q - current_q
        )
        self.q_table[state][action] = new_q
        
        self.epsilon = max(0.05, self.epsilon * 0.9999)
    
    def act(self, current_metrics, next_metrics):
        """
        Enhanced action selection and learning process.
        """
        state = self.get_state(current_metrics)
        next_state = self.get_state(next_metrics)
        
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in self.actions}
        
        action = self.choose_action(state)
        reward = compute_reward(current_metrics)
        
        
        if state != next_state:
            shaped_reward = reward * (1 + 0.1 * sum(abs(a - b) for a, b in zip(state, next_state)))
        else:
            shaped_reward = reward
        
        self.update(state, action, shaped_reward, next_state)
        return action, reward


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

# Initialize global agent with optimized parameters
global_rl_agent = RLAgent(actions, alpha=0.1, gamma=0.95, epsilon=0.15)

def get_rl_agent():
    """Return the global RL agent instance."""
    return global_rl_agent
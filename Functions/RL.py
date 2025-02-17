

import random
import numpy as np
from collections import defaultdict

def compute_reward(metrics):

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
       
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.q_table = defaultdict(lambda: {action: 0.0 for action in actions})
        self.visit_counts = defaultdict(lambda: defaultdict(int))
        self.total_updates = 0
        
    def get_state(self, metrics):
       
        clarity = metrics.get("clarity_score", 0.0)
        pitch = metrics.get("pitch_rate", 150)
        tone = metrics.get("tone", "Neutral")
        emotion = metrics.get("emotion", "Neutral")
      
        clarity_level = int(np.floor(clarity * 5))  
        

        pitch_ranges = [120, 140, 160, 180]
        pitch_level = sum(pitch > p for p in pitch_ranges)
        
        tone_map = {
            "Casual": 0, "Neutral": 1, "Persuasive": 2,
            "Formal": 3, "Confident": 4
        }
        tone_level = tone_map.get(tone, 1)
        
    
        emotion_map = {
            "Angry": 0, "Sad": 1, "Neutral": 2,
            "Excited": 3, "Happy": 4, "Calm": 5
        }
        emotion_level = emotion_map.get(emotion, 2)
        
        return (clarity_level, pitch_level, tone_level, emotion_level)
    
    def choose_action(self, state):
      
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


global_rl_agent = RLAgent(actions, alpha=0.1, gamma=0.95, epsilon=0.15)

def get_rl_agent():
    return global_rl_agent
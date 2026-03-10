# Целевые значения черт по кластерам (средние значения из mean_psychometric_clustering).
# Используются для выбора модификатора интенсивности по совпадению с участником.
trait_target_values = {
    0: {"openness": 25.71, "conscientiousness": 70.88, "extraversion": 36.23, "agreeableness": 55.24, "neuroticism": 45.32},
    1: {"openness": 47.85, "conscientiousness": 78.84, "extraversion": 75.02, "agreeableness": 68.52, "neuroticism": 17.36},
    2: {"openness": 45.63, "conscientiousness": 37.18, "extraversion": 69.48, "agreeableness": 38.64, "neuroticism": 44.31},
    3: {"openness": 40.39, "conscientiousness": 26.81, "extraversion": 22.13, "agreeableness": 39.93, "neuroticism": 78.77},
}

traits = {
    0: {
      "openness": "You prefer practical, concrete, and familiar ideas over abstract, unusual, or highly experimental ones.",
      "conscientiousness": "You are organized, dependable, and usually follow plans and responsibilities carefully.",
      "extraversion": "You are more reserved than outgoing and often prefer quieter settings or smaller social circles.",
      "agreeableness": "You are generally cooperative and considerate, while still able to assert your own needs when needed.",
      "neuroticism": "You experience a typical level of stress and emotional fluctuation, without being either highly reactive or unusually detached."
    },
    1: {
      "openness": "You show a balanced openness to new ideas, combining curiosity with a preference for what is practical and proven.",
      "conscientiousness": "You are highly reliable, structured, and motivated to complete tasks thoroughly and on time.",
      "extraversion": "You are socially energetic, talkative, and tend to seek interaction and group engagement.",
      "agreeableness": "You are warm, cooperative, and generally inclined toward trust, support, and social harmony.",
      "neuroticism": "You are emotionally stable and typically remain calm, even under pressure or uncertainty."
    },
    2: {
      "openness": "You have a moderate level of openness, balancing interest in novelty with comfort in familiar approaches.",
      "conscientiousness": "You tend to be flexible and spontaneous, with less focus on strict planning, order, or routine follow-through.",
      "extraversion": "You are outgoing and socially active, often gaining energy from interaction, discussion, and group settings.",
      "agreeableness": "You are more skeptical and competitive than accommodating, and may prioritize directness over maintaining harmony.",
      "neuroticism": "You experience a moderate level of emotional reactivity, with occasional stress or mood shifts."
    },
    3: {
      "openness": "You are somewhat conventional in your preferences, tending to favor familiar ideas over highly abstract or unconventional ones.",
      "conscientiousness": "You prefer spontaneity over structure and may find strict planning, organization, and consistency difficult to sustain.",
      "extraversion": "You are strongly introverted and usually prefer solitude or one-to-one interaction over larger social environments.",
      "agreeableness": "You may be somewhat critical, guarded, and less accommodating than average, often valuing independence over easy cooperation.",
      "neuroticism": "You are emotionally sensitive and prone to anxiety, tension, and noticeable mood variability."
    }
  }

system = {
    'role': "You are",
    'intensity_modifiers':  ['часто', 'иногда', 'редко'],
    'critic_internal': "You are",
    'template_structure': "TRAITS_FACETS_CRITIC",
    'task': f"""Ты должен ответить на опросник в соответсвии с тем как ты себя ощущаешь
        'select': {[
      {
         "id": 1,
         "text": "Very Inaccurate"
      },
      {
         "id": 2,
         "text": "Moderately Inaccurate"
      },
      {
         "id": 3,
         "text": "Neither Accurate Nor Inaccurate"
      },
      {
         "id": 4,
         "text": "Moderately Accurate"
      },
      {
         "id": 5,
         "text": "Very Accurate"
      }]
    }""",
    'response_format': "Ты должен двать ответы формата"
}
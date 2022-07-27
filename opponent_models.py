import random

def random_opp_model(gs, forward_model, player_id):
    actions = forward_model.generate_actions(gs, player_id)
    random_action = random.choice(actions)
    return random_action 

def attack_opp_model(gs, forward_model, player_id):
    available_attack_actions = []
    actions = forward_model.generate_actions(gs, player_id)
    for action in actions:
        if action.get_action_name() == "Attack":
            available_attack_actions.append(action)

    if available_attack_actions:
        attack_action = random.choice(available_attack_actions)
    else:
        attack_action = actions[-1]
    
    return attack_action

def do_nothing_model(gs, forward_model, player_id):
    actions = forward_model.generate_actions(gs, player_id)
    return actions[-1]



"""
Mapping from detected unit names (from YOLO detection) to their elixir cost.

The challenge is that:
- card_list.py has card names (e.g., "skeletons" for the card)
- label_list.py has unit names (e.g., "skeleton" for individual detected units)

This file bridges that gap, accounting for:
1. Cards that spawn multiple units (skeletons, bats, minion-horde, etc.)
2. Units that split (golem -> golemites, lava-hound -> lava-pups)
3. Spawner buildings (goblin-hut spawns spear-goblins)
4. Spell cards (fireball, zap, etc.)
5. Evolution variants

The elixir cost represents what the OPPONENT SPENT to get that unit on the field.
For split units (golemite, lava-pup), we use the parent's cost divided appropriately.
"""

from katacr.constants.label_list import unit_list, idx2unit, unit2idx

# Direct mapping from detected unit name to elixir cost
# This accounts for the actual elixir the opponent spent to deploy the unit
UNIT_TO_ELIXIR = {
    # ============ 1 ELIXIR CARDS ============
    'electro-spirit': 1,
    'fire-spirit': 1,
    'heal-spirit': 1,
    'ice-spirit': 1,
    'ice-spirit-evolution': 1,
    # Skeletons card (1 elixir) spawns 3-4 skeletons
    # We attribute 0.25 per skeleton (1/4) to encourage killing all
    'skeleton': 0.25,
    'skeleton-evolution': 0.25,
    
    # ============ 2 ELIXIR CARDS ============
    'barbarian-barrel': 2,  # The barrel itself (spawns 1 barbarian)
    # Bats card (2 elixir) spawns 5 bats -> 0.4 each
    'bat': 0.4,
    'bat-evolution': 0.4,
    'bomber': 2,
    'bomber-evolution': 2,
    'giant-snowball': 2,
    # Goblins card (2 elixir) spawns 3 goblins -> 0.67 each
    'goblin': 0.67,
    'ice-golem': 2,
    'rage': 2,
    # Spear goblins card (2 elixir) spawns 3 -> 0.67 each
    'spear-goblin': 0.67,
    'the-log': 2,
    'wall-breaker': 1,  # Wall breakers (2 elixir) spawns 2 -> 1 each
    'wall-breaker-evolution': 1,
    'zap': 2,
    'zap-evolution': 2,
    
    # ============ 3 ELIXIR CARDS ============
    # Archers card (3 elixir) spawns 2 -> 1.5 each
    'archer': 1.5,
    'archer-evolution': 1.5,
    'arrows': 3,
    'bandit': 3,
    'cannon': 3,
    'clone': 3,
    'dart-goblin': 3,
    'earthquake': 3,
    # Elixir golem (3 elixir) splits: big->2 mid->4 small
    'elixir-golem-big': 3,
    'elixir-golem-mid': 0.75,  # 2 spawn from big, worth ~1.5 total
    'elixir-golem-small': 0.375,  # 4 spawn from 2 mid, worth ~1.5 total
    'firecracker': 3,
    'firecracker-evolution': 3,
    'fisherman': 3,
    'goblin-barrel': 3,  # Spawns 3 goblins, barrel is the delivery
    # Goblin gang (3 elixir) spawns 3 goblins + 3 spear goblins
    # Goblins handled above, spear goblins handled above
    # Guards (3 elixir) spawns 3 -> 1 each
    'guard': 1,
    'ice-wizard': 3,
    'knight': 3,
    'knight-evolution': 3,
    'little-prince': 3,
    'mega-minion': 3,
    'miner': 3,
    'dirt': 0,  # Miner's dig animation, not a unit
    # Minions card (3 elixir) spawns 3 -> 1 each
    'minion': 1,
    'princess': 3,
    'royal-delivery': 3,  # Spawns 1 royal recruit
    'royal-ghost': 3,
    # Skeleton army (3 elixir) spawns 15 skeletons -> 0.2 each
    # (skeleton already mapped to 0.25 for regular skeletons card)
    'skeleton-barrel': 3,  # Spawns skeletons on death
    'tombstone': 3,  # Spawns skeletons over time
    'tornado': 3,
    
    # ============ 4 ELIXIR CARDS ============
    'baby-dragon': 4,
    'battle-healer': 4,
    'battle-ram': 4,  # Spawns 2 barbarians on death
    'battle-ram-evolution': 4,
    'bomb-tower': 4,
    'bomb': 0,  # Projectile, not a unit
    'dark-prince': 4,
    'electro-wizard': 4,
    'fireball': 4,
    'flying-machine': 4,
    'freeze': 4,
    'furnace': 4,  # Spawns fire spirits over time
    'goblin-cage': 4,  # Spawns goblin brawler
    'goblin-brawler': 4,  # Spawned by goblin cage
    'goblin-drill': 4,  # Spawns goblins
    'golden-knight': 4,
    'hog-rider': 4,
    'hunter': 4,
    'inferno-dragon': 4,
    'lumberjack': 4,
    'magic-archer': 4,
    'mighty-miner': 4,
    'mini-pekka': 4,
    'mortar': 4,
    'mortar-evolution': 4,
    'mother-witch': 4,
    'hog': 1,  # Cursed hog spawned by mother witch, minimal value
    'musketeer': 4,
    'night-witch': 4,  # Spawns bats
    'phoenix-big': 4,
    'phoenix-egg': 2,  # Half value, respawns phoenix
    'phoenix-small': 2,  # Respawned phoenix
    'poison': 4,
    # Skeleton dragons (4 elixir) spawns 2 -> 2 each
    'skeleton-dragon': 2,
    'skeleton-king': 4,
    'skeleton-king-bar': 0,  # UI element
    'skeleton-king-skill': 0,  # Ability visual
    'tesla': 4,
    'tesla-evolution': 4,
    'tesla-evolution-shock': 0,  # Ability visual
    'valkyrie': 4,
    'valkyrie-evolution': 4,
    # Zappies (4 elixir) spawns 3 -> 1.33 each
    'zappy': 1.33,
    
    # ============ 5 ELIXIR CARDS ============
    'archer-queen': 5,
    'balloon': 5,
    # Barbarians (5 elixir) spawns 5 -> 1 each
    'barbarian': 1,
    'barbarian-evolution': 1,
    'bowler': 5,
    'cannon-cart': 5,
    'electro-dragon': 5,
    'executioner': 5,
    'axe': 0,  # Executioner's axe, not a unit
    'giant': 5,
    'goblin-hut': 4,  # Spawns spear goblins
    'graveyard': 5,  # Spawns skeletons over area
    'inferno-tower': 5,
    # Minion horde (5 elixir) spawns 6 minions -> already counted as minion
    'monk': 5,
    'prince': 5,
    'ram-rider': 5,
    # Rascals (5 elixir) spawns 1 boy + 2 girls
    'rascal-boy': 2.5,
    'rascal-girl': 1.25,
    # Royal hogs (5 elixir) spawns 4 -> 1.25 each
    'royal-hog': 1.25,
    'witch': 5,  # Spawns skeletons
    'wizard': 5,
    
    # ============ 6 ELIXIR CARDS ============
    'barbarian-hut': 6,  # Spawns barbarians
    # Elite barbarians (6 elixir) spawns 2 -> 3 each
    'elite-barbarian': 3,
    'elixir-collector': 6,
    'giant-skeleton': 6,
    'goblin-giant': 6,  # Has spear goblins on back
    'goblin-ball': 0,  # Projectile
    'lightning': 6,
    'rocket': 6,
    'royal-giant': 6,
    'royal-giant-evolution': 6,
    'sparky': 6,
    'x-bow': 6,
    
    # ============ 7 ELIXIR CARDS ============
    'electro-giant': 7,
    'lava-hound': 7,
    # Lava pups (spawned when lava hound dies) - 6 pups
    'lava-pup': 1.17,  # 7/6 elixir each
    'mega-knight': 7,
    'pekka': 7,
    # Royal recruits (7 elixir) spawns 6 -> 1.17 each
    'royal-recruit': 1.17,
    'royal-recruit-evolution': 1.17,
    
    # ============ 8 ELIXIR CARDS ============
    'golem': 8,
    # Golemites (spawned when golem dies) - 2 golemites
    'golemite': 2,  # 4 elixir worth total (half of golem's value)
    
    # ============ 9 ELIXIR CARDS ============
    # Three musketeers handled via musketeer (3 * 4 = 12, but card costs 9)
    # Each musketeer from 3M is worth 3 elixir
    
    # ============ SPECIAL CARDS ============
    'mirror': 0,  # Copies last card, variable cost
    
    # ============ TOWERS (NOT CARDS) ============
    # Towers are worth a lot but handled separately in tower HP reward
    'king-tower': 0,
    'queen-tower': 0,
    'cannoneer-tower': 0,
    'dagger-duchess-tower': 0,
    'royal-guardian': 0,  # Part of tower defense
    
    # ============ UI/MISC ELEMENTS ============
    'tower-bar': 0,
    'king-tower-bar': 0,
    'dagger-duchess-tower-bar': 0,
    'bar': 0,
    'bar-level': 0,
    'clock': 0,
    'emote': 0,
    'text': 0,
    'elixir': 0,
    'selected': 0,
    'evolution-symbol': 0,
    'ice-spirit-evolution-symbol': 0,
}

def get_unit_elixir(unit_name: str) -> float:
    """
    Get the elixir value for a detected unit.
    
    Args:
        unit_name: The name of the unit as detected by YOLO
        
    Returns:
        The elixir cost/value of that unit. Returns 0 for unknown units.
    """
    return UNIT_TO_ELIXIR.get(unit_name, 0)

def get_unit_elixir_by_idx(cls_idx: int) -> float:
    """
    Get the elixir value for a detected unit by class index.
    
    Args:
        cls_idx: The class index from detection
        
    Returns:
        The elixir cost/value of that unit. Returns 0 for unknown units.
    """
    unit_name = idx2unit.get(cls_idx)
    if unit_name is None:
        return 0
    return get_unit_elixir(unit_name)


# Verify all units in label_list have a mapping
if __name__ == '__main__':
    from katacr.constants.label_list import unit_list
    
    missing = []
    for unit in unit_list:
        if unit not in UNIT_TO_ELIXIR:
            missing.append(unit)
    
    if missing:
        print(f"WARNING: {len(missing)} units missing from UNIT_TO_ELIXIR mapping:")
        for u in sorted(missing):
            print(f"  '{u}': 0,  # TODO: Add elixir cost")
    else:
        print(f"All {len(unit_list)} units have elixir mappings!")
    
    # Show non-zero elixir units
    print("\n=== Units with elixir value > 0 ===")
    for unit, elixir in sorted(UNIT_TO_ELIXIR.items(), key=lambda x: -x[1]):
        if elixir > 0:
            print(f"  {unit}: {elixir}")

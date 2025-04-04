from collections import Counter

def get_bias_stats(numbers):
    zones = {
        "Voisins": [22,18,29,7,28,12,35,3,26,0,32,15,19,4,21,2,25],
        "Tiers": [27,13,36,11,30,8,23,10,5,24,16,33],
        "Orphelins": [1,20,14,31,9,6,34,17]
    }
    freq = Counter(numbers)
    zone_hits = {zone: sum(freq[n] for n in nums) for zone, nums in zones.items()}
    return zone_hits
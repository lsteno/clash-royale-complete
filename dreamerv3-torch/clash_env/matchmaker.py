
import threading
import time
import subprocess
from pathlib import Path
import sys

class MatchMaker:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'): return
        self._devices = []
        self._ready_devices = set()
        self._reset_done = set()
        self._last_host_action = {}
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._initialized = True
        
        # Detect devices
        res = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        lines = res.stdout.strip().split('\n')[1:]
        self._devices = [l.split('\t')[0] for l in lines if l.strip() and 'offline' not in l]
        self._devices.sort()
        print(f"[MatchMaker] Thread-Safe Initialized with {len(self._devices)} devices: {self._devices}")

    def get_pair_partner(self, device_id):
        if device_id not in self._devices: return None
        idx = self._devices.index(device_id)
        partner_idx = idx + 1 if idx % 2 == 0 else idx - 1
        if partner_idx < len(self._devices):
            return self._devices[partner_idx]
        return None

    def get_device_index(self, device_id):
        if device_id not in self._devices:
            return None
        return self._devices.index(device_id)

    def is_joiner(self, device_id):
        idx = self.get_device_index(device_id)
        return idx is not None and idx % 2 == 1

    def mark_host_action(self, device_id):
        idx = self.get_device_index(device_id)
        if idx is None or idx % 2 == 1:
            return
        partner_id = self.get_pair_partner(device_id)
        if partner_id is None:
            return
        ts = time.time()
        with self._lock:
            self._last_host_action[partner_id] = ts

    def get_last_host_action(self, device_id):
        with self._lock:
            return self._last_host_action.get(device_id)

    def mark_ready(self, device_id):
        with self._lock:
            self._ready_devices.add(device_id)
            self._cv.notify_all()

    def try_joint_reset(self, device_id):
        partner_id = self.get_pair_partner(device_id)
        if partner_id is None: return True

        with self._lock:
            # Check if partner already did the reset for us
            if device_id in self._reset_done:
                self._reset_done.remove(device_id)
                return True

            if device_id not in self._ready_devices or partner_id not in self._ready_devices:
                return False
            
            # Both are ready. FIRST one triggers reset.
            idx = self._devices.index(device_id)
            partner_idx = self._devices.index(partner_id)
            
            if idx < partner_idx:
                print(f"[MatchMaker] Both ready. Device {device_id} performing JOINT RESET for pair...")
                from clash_env.coordinator import SelfPlayCoordinator
                coord = SelfPlayCoordinator([device_id, partner_id])
                coord.reset()
                
                # Cleanup
                self._ready_devices.remove(device_id)
                self._ready_devices.remove(partner_id)
                
                # Notify partner
                self._reset_done.add(partner_id)
                self._cv.notify_all()
                return True
            else:
                return False

    def wait_and_reset(self, device_id, env_obj):
        """Blocking version."""
        self.mark_ready(device_id)
        while True:
            if self.try_joint_reset(device_id):
                break
            time.sleep(1.0)

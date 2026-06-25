import math


class CentroidTracker:
    def __init__(self, max_distance=80, max_missing_frames=10):
        
        self.next_id = 0
        self.objects = {}         
        self.missing_frames = {}  
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames

    def update(self, centroids):
        
        if len(centroids) == 0:
            for obj_id in list(self.missing_frames.keys()):
                self.missing_frames[obj_id] += 1
                if self.missing_frames[obj_id] > self.max_missing_frames:
                    del self.objects[obj_id]
                    del self.missing_frames[obj_id]
            return self.objects

        if len(self.objects) == 0:
            for c in centroids:
                self._register(c)
            return self.objects

        existing_ids = list(self.objects.keys())
        existing_centroids = list(self.objects.values())

        pairs = []
        for i, ec in enumerate(existing_centroids):
            for j, nc in enumerate(centroids):
                d = self._dist(ec, nc)
                pairs.append((d, i, j))
        pairs.sort(key=lambda x: x[0])

        assigned_existing = set()
        assigned_new = set()

        for d, i, j in pairs:
            if i in assigned_existing or j in assigned_new:
                continue
            if d > self.max_distance:
                continue
            obj_id = existing_ids[i]
            self.objects[obj_id] = centroids[j]
            self.missing_frames[obj_id] = 0
            assigned_existing.add(i)
            assigned_new.add(j)

        for j, c in enumerate(centroids):
            if j not in assigned_new:
                self._register(c)

        for i, obj_id in enumerate(existing_ids):
            if i not in assigned_existing:
                self.missing_frames[obj_id] = self.missing_frames.get(obj_id, 0) + 1
                if self.missing_frames[obj_id] > self.max_missing_frames:
                    del self.objects[obj_id]
                    del self.missing_frames[obj_id]

        return self.objects

    def _register(self, centroid):
        self.objects[self.next_id] = centroid
        self.missing_frames[self.next_id] = 0
        self.next_id += 1

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])


class LineCrossingCounter:
    def __init__(self, orientation="vertical", position_ratio=0.5):
        
        self.orientation = orientation
        self.position_ratio = position_ratio
        self.prev_positions = {}
        self.counts = {"in": 0, "out": 0}

    def get_line_coordinate(self, frame_width, frame_height):
        if self.orientation == "vertical":
            return int(frame_width * self.position_ratio)
        else:
            return int(frame_height * self.position_ratio)

    def update(self, tracked_objects, frame_width, frame_height):
        
        line_coord = self.get_line_coordinate(frame_width, frame_height)
        current_ids = set(tracked_objects.keys())

        for obj_id, (cx, cy) in tracked_objects.items():
            coord = cx if self.orientation == "vertical" else cy

            if obj_id in self.prev_positions:
                prev_coord = self.prev_positions[obj_id]

                if prev_coord < line_coord <= coord:
                    self.counts["in"] += 1
                elif prev_coord > line_coord >= coord:
                    self.counts["out"] += 1

            self.prev_positions[obj_id] = coord

        stale_ids = set(self.prev_positions.keys()) - current_ids
        for obj_id in stale_ids:
            del self.prev_positions[obj_id]

        return dict(self.counts)

    def reset(self):
        self.prev_positions = {}
        self.counts = {"in": 0, "out": 0}
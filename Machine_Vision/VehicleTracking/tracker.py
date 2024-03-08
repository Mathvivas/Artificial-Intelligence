import math

class Tracker:
    def __init__(self):
        # Store the center of objects
        self.center_points = {}
        # Counter of objects
        self.id_count = 0

    def update(self, box):
        # Object's boxes and ids
        objects_bbs_ids = []

        # Get the center of an object
        for coord in box:
            x, y, w, h = coord
            cx = (x + w) // 2
            cy = (y + h) // 2

            # Check if the object was already detected
            detected = False
            for id, pt in self.center_points.items():
                # The Euclidian norm is the distance from the origin to the coordinates given.
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    detected = True
                    break

            # Assign ID to new object detected
            if detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Remove IDs not used anymore
        new_center_points = {}
        for object_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = object_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
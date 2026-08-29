import sys
import os

import pygame
import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pygame


from ai.perception import RobustLanePerception


def main():

    print()
    print("=" * 70)
    print("ROBUST-LANENET V3 INFERENCE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # PYGAME
    # --------------------------------------------------------

    pygame.init()

    screen = pygame.display.set_mode(
        (640, 360)
    )

    screen.fill(
        (80, 80, 80)
    )

    pygame.display.flip()

    # --------------------------------------------------------
    # PERCEPTION
    # --------------------------------------------------------

    perception = RobustLanePerception()

    # --------------------------------------------------------
    # CAPTURE SCREEN
    # --------------------------------------------------------

    raw = pygame.surfarray.array3d(
        screen
    )

    # Pygame:
    # width × height × RGB

    # Convert to:
    # height × width × RGB

    raw = raw.transpose(
        (1, 0, 2)
    )

    # Convert RGB → BGR

    import cv2

    image = cv2.cvtColor(
        raw,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    lanes = perception.predict(
        image
    )

    print()
    print("Predicted lanes:")

    for i, x in enumerate(lanes):

        print(
            f"Lane {i}: "
            f"{x:.2f} px"
        )

    print()
    print("=" * 70)
    print("V3 INFERENCE TEST COMPLETE")
    print("=" * 70)

    pygame.quit()


if __name__ == "__main__":
    main()
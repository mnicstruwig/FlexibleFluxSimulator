"""
Utility functions for rendering the microgenerator in ascii.
"""
import numpy as np

empty_coil = "│        │"
coil_bottom = "└─┼────┼─┘"
coil_top = "┌─┼────┼─┐"
magnet_symbol = "█"
spacer_symbol = "▒"
empty_tube = "  │    │"
bottom = "  └────┘"
top = "  ┌────┐"
empty = "        "


def is_magnet(z, l_m_mm, m, l_hover, l_mcd_mm):
    if 0 <= z <= l_m_mm:  # Fixed magnet
        return True

    l_base_center = l_hover + l_m_mm / 2
    for n_m in range(1, m + 1):
        lower_limit = l_base_center + (n_m - 1) * l_mcd_mm - l_m_mm / 2
        upper_limit = l_base_center + (n_m - 1) * l_mcd_mm + l_m_mm / 2

        if lower_limit < z <= upper_limit:  # Magnet assembly
            return True
    return False


def is_coil(z, l_c, c, l_center, l_ccd_mm):
    for n_c in range(1, c + 1):
        lower_limit = (l_center + (n_c - 1) * l_ccd_mm) - l_c / 2
        upper_limit = (l_center + (n_c - 1) * l_ccd_mm) + l_c / 2
        if lower_limit < z <= upper_limit:
            return True
    return False


def is_tube(z, l_L):
    if z < l_L:
        return True
    return False


def round_to_nearest(x, base=5):
    result = base * round(x / base)
    if result == 0:
        return base
    return result


def make_device_definition(
    step, m, l_m_mm, l_mcd_mm, l_hover, c, l_c_mm, l_ccd_mm, l_center, l_L
):
    l_m_mm = round_to_nearest(l_m_mm, step)
    l_mcd_mm = round_to_nearest(l_mcd_mm, step)
    l_hover = round_to_nearest(l_hover, step)
    l_c_mm = round_to_nearest(l_c_mm, step)
    l_ccd_mm = round_to_nearest(l_ccd_mm, step)
    l_center = round_to_nearest(l_center, step)
    l_L = round_to_nearest(l_L, step)

    device_defn = []
    for z in np.arange(step, 200, step):
        magnet = is_magnet(z, l_m_mm, m, l_hover, l_mcd_mm)
        coil = is_coil(z, l_c_mm, c, l_center, l_ccd_mm)
        tube = is_tube(z, l_L)
        device_defn.append((z, magnet, coil, tube))
    return device_defn


def create_drawing_config(device_defn):
    device = []
    device.append(bottom)
    in_coil = False
    in_tube = True

    # We draw from bottom of the device to the top
    for i, defn in enumerate(device_defn):
        z, magnet, coil, tube = defn

        line = empty  # Start with an empty line

        if not coil:  # If not in a coil
            if in_coil:  # But we were in a coil
                in_coil = False  # Deactivate coil drawing
                line = coil_top  # And draw the top of the coil

        if coil:  # If we're in a coil
            if not in_coil:  # But we weren't
                line = coil_bottom  # Draw the bottom of the coil
                in_coil = True
            elif in_coil:  # If we were in a coil
                line = empty_coil  # Continue drawing the coil

        if tube:  # If we're in the tube
            line = list(line)  # Draw the tube lines
            line[2] = "│"
            line[7] = "│"
            line = "".join(line)
        else:  # If we're not in the tube
            if in_tube:  # But we were on the previous step
                in_tube = False  # Deactivate tube drawing

                if "─" in line:  # If we're intersecting a coil
                    line = list(line)
                    line[2] = "┌"  # Just close off the tube
                    line[7] = "┐"
                    line = "".join(line)
                else:  # If we're not intersecting anything
                    line = top  # Cap it off normally.

        if magnet:  # If we're in a magnet
            line = list(line)
            line[3] = magnet_symbol  # Paint in the magnet
            line[4] = magnet_symbol
            line[5] = magnet_symbol
            line[6] = magnet_symbol
            line = "".join(line)

        device.insert(0, line)  # Draw our line
    return device


def find_magnet_start_end_indices(device):
    in_magnet = False
    start_idxs = []
    end_idxs = []
    for i, row in enumerate(device):  # Find the cells between magnets
        if magnet_symbol in row and not in_magnet:
            in_magnet = True
            start_idxs.append(i)

        if magnet_symbol not in row and in_magnet:
            end_idxs.append(i)
            in_magnet = False

    # The last indexes are the fixed magnet at the bottom, which we want to ignore
    start_idxs = start_idxs[:-1]
    end_idxs = end_idxs[:-1]

    return start_idxs, end_idxs


def add_spacers(device):
    start_idxs, end_idxs = find_magnet_start_end_indices(device)
    # For each row between when the spacer starts and stops...
    for spacer_start, spacer_end in zip(end_idxs, start_idxs[1:]):
        for row_index in range(spacer_start, spacer_end):  # Fill in the spacer symbol.
            line = list(device[row_index])
            line[3] = spacer_symbol
            line[4] = spacer_symbol
            line[5] = spacer_symbol
            line[6] = spacer_symbol
            line = "".join(line)
            device[row_index] = line
    return device


def design_is_valid(device_defn):
    # If any magnet or coil occurs when we're not inside the tube
    # then we don't have a valid design
    for defn in device_defn:
        _, magnet, coil, in_tube = defn

        if not in_tube:
            if magnet or coil:
                return False
    return True


def add_validity_checks(device, device_defn):
    if design_is_valid(device_defn):  # Truncate
        for i, row in enumerate(device):
            if top.strip() in row:
                device = device[i:]
                break
    else:
        # First truncate to highest offender
        for i, row in enumerate(device):
            if (magnet_symbol in row) or (spacer_symbol in row) or ("┼" in row):
                device = device[i:]
                break

        for i, row in enumerate(device):
            if "┌" == row[2]:
                device[i] = device[i] + "<--- !!"
    return device


def add_line_numbers(device, device_defn):
    step = device_defn[1][0] - device_defn[0][0]
    z = 0
    for i, row in list(enumerate(device))[::-1]:  # We want to loop in reverse
        device[i] = row + "".join([" "] * (17 - len(row))) + str(z)
        z = z + step
    return device


def paint_device(
    step, m, l_m_mm, l_mcd_mm, l_hover, c, l_c_mm, l_ccd_mm, l_center, l_L
):
    device_defn = make_device_definition(
        step=step,
        m=m,
        l_m_mm=l_m_mm,
        l_mcd_mm=l_mcd_mm,
        l_hover=l_hover,
        c=c,
        l_c_mm=l_c_mm,
        l_ccd_mm=l_ccd_mm,
        l_center=l_center,
        l_L=l_L,
    )
    device = create_drawing_config(device_defn)
    device = add_spacers(device)
    device = add_validity_checks(device, device_defn)
    device = add_line_numbers(device, device_defn)

    print()
    for x in device:
        print(x)

import numpy as np

empty_coil =  "│        │"
coil_bottom = "└─┼────┼─┘"
coil_top =    "┌─┼────┼─┐"
magnet_symbol = "█"
spacer_symbol = "▒"
empty_tube = "  │    │"
bottom = "  └────┘"
top = "  ┌────┐"
empty = "        "


def is_magnet(z, l_m_mm, m, l_hover, l_mcd_mm):  # do for single magnet only, for now
    if 0 <= z <= l_m_mm:  # Fixed magnet case
        return True

    l_base_center = l_hover + l_m_mm / 2
    for n_m in range(1, m+1):
        lower_limit = (l_base_center + (n_m - 1) * l_mcd_mm - l_m_mm/2)
        upper_limit = (l_base_center + (n_m - 1) * l_mcd_mm + l_m_mm/2)

        if lower_limit < z <= upper_limit:  # Magnet assembly
            return True
    return False


def is_coil(z, l_c, c, l_center, l_ccd_mm):
    for n_c in range(1, c + 1):
        lower_limit = (l_center + (n_c - 1)*l_ccd_mm) - l_c / 2
        upper_limit = (l_center + (n_c - 1)*l_ccd_mm) + l_c / 2
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



# Device parameters
m = 1
c = 2
l_m_mm = 10
l_mcd_mm = round_to_nearest(30)
l_c = round_to_nearest(10)
l_ccd_mm = round_to_nearest(30)
l_center = round_to_nearest(60)
l_hover = 40
l_L = 100

# First lets make a device definition
step = 5
device_defn = []
for z in np.arange(step, 200, step):
    magnet = is_magnet(z, l_m_mm, m, l_hover, l_mcd_mm)
    coil = is_coil(z, l_c, c, l_center, l_ccd_mm)
    tube = is_tube(z, l_L)
    device_defn.append((z, magnet, coil, tube))

# Draw
device = []
device.append(bottom)
in_coil = False
in_assembly = False
in_tube = True
for i, defn in enumerate(device_defn):
    z, magnet, coil, tube = defn
    line = empty
    # if tube:
    #     line = empty_tube
    # else:
    #     if in_tube:
    #         in_tube = False
    #         line = top
    #     else:
    #         line = empty

    if not coil:
        if in_coil:
            in_coil = False
            line = coil_top

    if coil:
        if not in_coil:
            line = coil_bottom
            in_coil = True
        elif in_coil:
            line = empty_coil

    if tube:
        line = list(line)
        line[2] = "│"
        line[7] = "│"
        line = ''.join(line)
    else:
        if in_tube:
            in_tube = False
            if '─' in line:  # If we're intersecting a coil
                line = list(line)
                line[2] = "┌"  # Just close off the tube
                line[7] = "┐"
                line = ''.join(line)
            else:  # If we're not intersecting anything
                line = top  # Cap it off normally.
        else:
            line = empty

    if magnet:
        line = list(line)
        line[3] = magnet_symbol
        line[4] = magnet_symbol
        line[5] = magnet_symbol
        line[6] = magnet_symbol
        line = ''.join(line)

    line_length = len(line)
    spaces = 14 - line_length

#    line = line + ''.join([" "]*spaces) + str(z)
    device.insert(0, line)
#device.insert(0, top)

# Fill in spacers afterwards
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

# The last indexes are the floating magnet, which we want to ignore
start_idxs = start_idxs[:-1]
end_idxs = end_idxs[:-1]

# For each row between when the spacer starts and stops...
for spacer_start, spacer_end in zip(end_idxs, start_idxs[1:]):
    for row_index in range(spacer_start, spacer_end):  # Fill in the spacer symbol.
        line = list(device[row_index])
        line[3] = spacer_symbol
        line[4] = spacer_symbol
        line[5] = spacer_symbol
        line[6] = spacer_symbol
        line = ''.join(line)
        device[row_index] = line


def design_is_valid(device_defn):
    # If any magnet or coil occurs when we're not inside the tube
    # then we don't have a valid design
    for defn in device_defn:
        _, magnet, coil, in_tube = defn

        if not in_tube:
            if magnet or coil:
                return False
    return True

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

z = 0
for i, row in list(enumerate(device))[::-1]:  # We want to loop in reverse
    device[i] = row + ''.join([" "] * (17 - len(row))) + str(z)
    z = z + step

print("\n")
for x in device:
    print(x)

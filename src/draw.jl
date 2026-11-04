"""
    draw_tile(k, offset, sites; radius=0.15)

Draw one cluster in the current Luxor drawing. The bits of `k - 1`, least
significant first, follow the order of `sites`: 0 is spin +1 (black),
1 is spin -1 (white). Coordinates and radius are in lattice units.
"""
function draw_tile(k::Int, offset, sites; radius=0.15)
    spins = 1 .- 2 .* digits(k - 1; base=2, pad=length(sites))
    for (spin, site) in zip(spins, sites)
        position = Point((offset .+ site)...)
        sethue(spin == 1 ? "black" : "white")
        circle(position, radius, :fill)
    end
end
 
"""
    draw_spin_configuration(filename, conf, lattice_vectors, sites; size=200)

Save the cluster labels `conf[x, y]` as a square image of `size` pixels per
side. Use the bit encoding of `draw_tile`. The two lattice vectors place
neighboring clusters; `sites` gives spin positions relative to each cluster.
The drawing is centered and scaled to fit, with positive y pointing up.
"""
function draw_spin_configuration(filename::String, conf::Matrix{Int}, lattice_vectors, sites; size=200)
    @assert size > 0 "Image size must be positive."
    @assert !isempty(conf) && !isempty(sites) "Configurations and sites must be nonempty."
    Lx, Ly = Base.size(conf)
    a1, a2 = lattice_vectors

    # Choose a circle radius smaller than the spacing between sites.
    spacing = min(norm(a1), norm(a2))
    for i in eachindex(sites), j in 1:i-1
        distance = norm(sites[i] .- sites[j])
        if distance > 0
            spacing = min(spacing, distance)
        end
    end
    radius = 0.4*spacing

    # The outermost clusters contain all the coordinate extrema.
    positions = [(x - 1) .* a1 .+ (y - 1) .* a2 .+ site
                 for x in (1, Lx), y in (1, Ly), site in sites]
    xmin, xmax = extrema(p[1] for p in positions)
    ymin, ymax = extrema(p[2] for p in positions)
    padding = 2 * radius
    drawing_scale = size / max(xmax - xmin + 2 * padding, ymax - ymin + 2 * padding)

    Drawing(size, size, filename)
    background("white")
    origin()
    Luxor.scale(drawing_scale, -drawing_scale)
    translate(-(xmin + xmax) / 2, -(ymin + ymax) / 2)

    for x in 1:Lx, y in 1:Ly
        offset = (x - 1) .* a1 .+ (y - 1) .* a2
        draw_tile(conf[x, y], offset, sites; radius)
    end
    finish()
    return filename
end

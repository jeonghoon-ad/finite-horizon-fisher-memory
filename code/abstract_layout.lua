-- Typesetting only: preserve every existing manuscript block.
function Pandoc(doc)
  if not FORMAT:match("latex") then return doc end
  local out, active, count = {}, false, 0
  for _, block in ipairs(doc.blocks) do
    if block.t == "Header" and active then
      table.insert(out, pandoc.RawBlock("latex", "\\end{preprintabstract}"))
      active = false
    end
    table.insert(out, block)
    if block.t == "Header" and pandoc.utils.stringify(block.content) == "Abstract" then
      table.insert(out, pandoc.RawBlock("latex", "\\begin{preprintabstract}"))
      active, count = true, count + 1
    end
  end
  assert(count == 1 and not active, "Expected one complete Abstract section")
  return pandoc.Pandoc(out, doc.meta)
end

"""
Two chunking approaches were applied to the same dataset:

Fixed-Size Chunking

Documents were split into uniform chunks of approximately 500 characters.

Chunking was based solely on size, without considering sentence or topic boundaries.

Semantic Chunking

Documents were split using embedding-based semantic similarity.

Chunk boundaries were determined by topic shifts and meaning changes rather than character count.

Both chunk types were indexed and queried using the same retrieval and LLM setup to ensure a fair comparison.

Retrieval Quality Comparison
Fixed-Size Chunking

Frequently split sentences and paragraphs.

Retrieved chunks often contained incomplete thoughts.

Important contextual details were sometimes missing or cut off.

Observed Issue:
Retrieved results occasionally started or ended mid-sentence, reducing usefulness for downstream answer generation.

Semantic Chunking

Preserved complete ideas and topic boundaries.

Retrieved chunks were more self-contained and meaningful.

Provided full explanations without requiring additional context.

Observed Benefit:
Chunks consistently returned coherent, actionable information relevant to the query.

Testing Across Document Types
Document Type	Fixed Chunking	Semantic Chunking
Long guides / reports	Poor coherence	Strong coherence
Educational content	Fragmented	Well-structured
Narrative text	Often broken	Fully preserved
Short structured docs	Acceptable	Slight improvement

Conclusion:
Semantic chunking showed the greatest advantage for long-form and narrative documents where context flows across paragraphs.

Impact on Answer Coherence

Fixed Chunking:
Generated answers sometimes lacked clarity due to missing context.

Semantic Chunking:
Answers were more complete, accurate, and easier to understand because the retrieved chunks contained full ideas.

Key Insight:
Better chunk coherence directly improved answer quality, even when using the same language model.

Trade-Off Summary
Aspect	Fixed Chunking	Semantic Chunking
Speed	Faster	Slower
Setup complexity	Simple	Moderate
Token efficiency	Lower	Higher
Retrieval quality	Inconsistent	High
Answer coherence	Variable	Strong
Conclusion

Semantic chunking significantly improves retrieval quality and answer coherence by preserving meaning and context within chunks. Although it introduces additional computational cost due to embedding calculations, the improvement in response quality makes it a valuable technique for applications where accuracy and completeness are critical.
"""
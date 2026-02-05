# Case Sensitivity Enhancement Plan for Alias Rules

## 1. Problem Statement

### Current Issues

1. **Duplicate Rules for Case Variations**: The current `alias_rules.csv` contains many duplicate rules to handle case variations. For example:
   - Lines 7-9: `애플,Apple`, `애플,APPLE`, `애플,apple` (3 rows)
   - Lines 21-25: `아이폰,iPhone`, `아이폰,IPHONE`, `아이폰,iphone`, `IPHONE,iPhone`, `iPhone,iphone` (5 rows)
   - Similar patterns exist for 40+ terms (Samsung, Galaxy, Pro, Max, etc.)

2. **Increased Cascading Depth**: To handle case variations like `IPHONE` → `아이폰` → `iPhone`, the `max_depth` parameter had to be increased from 3 to 7, increasing processing time.

3. **Maintenance Burden**: Adding a new alias requires creating multiple rows for each case variation.

### Current Duplicate Count Analysis

| Category | Terms | Rows per Term | Total Duplicate Rows |
|----------|-------|---------------|---------------------|
| Brands | 애플, 삼성, 샤오미, 구글 | 3 | 12 |
| Devices | 아이폰, 아이패드, 갤럭시, 워치, 노트, 탭, 레드미 | 3-5 | 25+ |
| Colors | 블랙, 화이트, 블루, 레드, 그린, 골드, 실버 | 3 | 21 |
| Size | 울트라, 맥스, 미니, 플립, 폴드, 라이트 | 3 | 18 |
| Service | 케어, 플러스, 프로, 프리미엄, 스마트, 패스 | 3 | 18 |
| Tech | 웹, 와이파이, 앱, 데이터, 비디오, 오디오, 클라우드 | 3-4 | 25+ |
| **Total** | | | **~120 rows** |

**Potential Reduction**: ~120 duplicate rows → ~40 single rows = **67% reduction**

---

## 2. Proposed Solution

### 2.1 New CSV Schema

Add two new columns to `alias_rules.csv`:

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `case_1` | string | `S` (sensitive), `I` (insensitive) | Case sensitivity for `alias_1` matching |
| `case_2` | string | `S` (sensitive), `I` (insensitive) | Case sensitivity for `alias_2` generation |

### 2.2 Updated CSV Structure

```csv
alias_1,alias_2,category,description,direction,type,case_1,case_2
```

### 2.3 Example Transformation

**Before (3 rows):**
```csv
애플,Apple,brand,IT 기업,B,partial
애플,APPLE,brand,IT 기업,B,partial
애플,apple,brand,IT 기업,B,partial
```

**After (1 row):**
```csv
애플,Apple,brand,IT 기업,B,partial,S,I
```

Meaning:
- `case_1=S`: Match `애플` exactly (Korean, case doesn't apply)
- `case_2=I`: `Apple` matches `Apple`, `APPLE`, `apple` (case-insensitive)

### 2.4 Case Sensitivity Logic

| case_1 | case_2 | Matching Behavior |
|--------|--------|-------------------|
| S | S | Exact match on both sides |
| S | I | Exact alias_1 match, case-insensitive alias_2 generation |
| I | S | Case-insensitive alias_1 match, exact alias_2 generation |
| I | I | Case-insensitive on both sides |

---

## 3. Implementation Plan

### Phase 1: Update Data Schema

1. **Add new columns to `alias_rules.csv`**
   - Add `case_1` and `case_2` columns with default value `S` (sensitive)
   - Set `case_2=I` for all English terms that have case duplicates
   - Remove duplicate rows

2. **Migrate existing rules**
   - Analyze each rule group
   - Consolidate case duplicates into single rules with `I` flag
   - Preserve rules that truly require case-sensitive matching

### Phase 2: Update Processing Logic

1. **Modify `apply_cascading_alias_rules()` in `item_data_loader.py`**

   ```python
   # Current logic (line 407-413):
   if alias_type == 'exact':
       matched = (current_item == alias_from)
   else:
       matched = (alias_from in current_item)

   # New logic:
   if alias_type == 'exact':
       if case_1 == 'I':
           matched = (current_item.lower() == alias_from.lower())
       else:
           matched = (current_item == alias_from)
   else:  # partial
       if case_1 == 'I':
           matched = (alias_from.lower() in current_item.lower())
       else:
           matched = (alias_from in current_item)
   ```

2. **Handle case_2 for alias generation**

   When generating aliases with `case_2=I`, generate all case variants:
   - Original case
   - Lowercase
   - Uppercase
   - Title case (optional)

3. **Update `load_alias_rules()`**
   - Parse new `case_1` and `case_2` columns
   - Set default values for backward compatibility

### Phase 3: Reduce max_depth

1. **Test with reduced max_depth**
   - With case-insensitive matching, fewer cascading iterations needed
   - Potentially reduce `max_depth` from 7 back to 3-4

2. **Performance validation**
   - Compare processing time before/after
   - Verify alias generation correctness

---

## 4. Migration Strategy

### 4.1 Backward Compatibility

- Default `case_1=S` and `case_2=S` maintains current behavior
- Existing rules without new columns work unchanged

### 4.2 Step-by-Step Migration

1. **Add columns with defaults** (no behavior change)
2. **Update high-impact rules first**:
   - iPhone/아이폰 variations (5 rows → 1)
   - Galaxy/갤럭시 variations (3 rows → 1)
   - Common English terms (Pro, Max, Plus, etc.)
3. **Remove duplicate rows**
4. **Test alias generation**
5. **Reduce max_depth** if tests pass

### 4.3 Rollback Plan

- Keep backup of original `alias_rules.csv`
- Can revert by removing new columns and restoring duplicates

---

## 5. Expected Benefits

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Rule count | ~190 rows | ~70 rows | 63% reduction |
| max_depth | 7 | 3-4 | 43-57% reduction |
| Maintenance | Add 3+ rows per term | Add 1 row per term | 66% less work |
| Processing time | Higher | Lower | TBD after implementation |

---

## 6. Files to Modify

1. **`data/alias_rules.csv`**
   - Add `case_1` and `case_2` columns
   - Consolidate duplicate rules

2. **`services/item_data_loader.py`**
   - Update `load_alias_rules()` to parse new columns
   - Update `apply_cascading_alias_rules()` for case-insensitive matching

3. **`tests/generate_alias_list.py`**
   - Update test utility to support new schema

---

## 7. Test Cases

### 7.1 Validation Tests

1. `iPhone 17` should generate same aliases as before
2. `iphone 17` should generate same aliases as `iPhone 17`
3. `IPHONE 17` should generate same aliases as `iPhone 17`
4. Korean terms (아이폰) should work unchanged

### 7.2 Edge Cases

1. Mixed case input: `IPhone 17` → should match `iPhone` rules
2. Korean + English mix: `아이폰 Pro` → should handle both
3. Build type aliases with case variations

---

## 8. Implementation Status ✅

All phases completed successfully!

### Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Rule count | 188 rows | 81 rows | **57% reduction** |
| max_depth required | 7 | 3 | **57% reduction** |

### Validation Test Results

```
iPhone 17 → 10 aliases (including 아이폰 17, 아이폰 신제품, etc.)
iphone 17 → 10 aliases (same as iPhone 17) ✅
IPHONE 17 → 10 aliases (same as iPhone 17) ✅
```

### Files Modified

1. `data/alias_rules.csv` - Added case_1, case_2 columns, consolidated duplicates
2. `services/item_data_loader.py` - Updated matching logic for case sensitivity
3. `tests/generate_alias_list.py` - Added --max-depth option
4. `data/alias_rules_backup.csv` - Backup of original rules

---

## 9. Design Decisions (Resolved)

1. **Title case handling**: Should `Iphone` match `iPhone` with `case_2=I`?
   - ✅ **Yes**, use `.lower()` comparison for all case-insensitive matching

2. **Korean case**: Korean doesn't have case variations, always use `S`?
   - ✅ **User will set the rule** for each case in `alias_rules.csv` explicitly

3. **Partial match case handling**: For partial matches, should case sensitivity apply to the substring match only?
   - ✅ **Yes**, only the matching portion is case-insensitive

4. **Default behavior**: What should be the default when columns are missing?
   - ✅ **`S` (sensitive)** for backward compatibility
